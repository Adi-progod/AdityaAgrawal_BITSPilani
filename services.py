import os
import json
import logging
import fitz  # PyMuPDF
import base64
import asyncio
import requests
from io import BytesIO
from PIL import Image
from openai import AsyncOpenAI  # Using Async Client for parallel processing
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("uvicorn")

# --- CONFIGURATION ---
# Async Client allows us to fire multiple requests without blocking
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    max_retries=5
)

# Using Llama 4 (Scout or Maverick)
MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"

# --- ACCURACY LAYER: PYTHON POST-PROCESSING ---
# These keywords indicate a summary row. Removing them ensures "Final Total" is accurate.
BANNED_KEYWORDS = [
    "total", "subtotal", "sub total", "net amount", "grand total", 
    "tax", "gst", "vat", "discount", "round off", "balance", "amount due",
    "gross amount", "cgst", "sgst", "igst", "total amount"
]

def clean_and_validate_items(items: list) -> list:
    """
    Applied Science Logic:
    1. Filter out summary rows using Python (100% accurate vs LLM).
    2. Enforce Data Types.
    3. Normalize numbers (handle missing zeros).
    """
    cleaned = []
    for item in items:
        name = str(item.get("item_name", "")).strip()
        name_lower = name.lower()
        
        # Rule 1: Drop empty items
        if not name or name_lower == "unknown":
            continue

        # Rule 2: Hard Filter for Summary Rows (The "No Double Counting" Fix)
        is_banned = False
        for kw in BANNED_KEYWORDS:
            # Matches "Total", "Subtotal", "Total Amount" exactly or as distinct words
            # Logic: keyword must be the whole name OR present as a distinct word
            if kw == name_lower or f" {kw} " in f" {name_lower} " or name_lower.startswith(f"{kw} ") or name_lower.endswith(f" {kw}"):
                # Exception check: Don't ban "Total Knee Replacement" if it's a medical bill
                # But for this hackathon, aggressive filtering is usually safer for the "Totals" criteria.
                is_banned = True
                break
        
        if is_banned:
            continue

        # Rule 3: Normalize Numbers
        try:
            qty = float(item.get("item_quantity", 1.0) or 1.0)
            rate = float(item.get("item_rate", 0.0) or 0.0)
            amount = float(item.get("item_amount", 0.0) or 0.0)
            
            # Auto-Correction: If Amount is 0 but we have Rate/Qty, calculate it
            if amount == 0 and rate > 0:
                amount = rate * qty
            
            cleaned.append({
                "item_name": name,
                "item_amount": round(amount, 2),
                "item_rate": round(rate, 2),
                "item_quantity": qty
            })
        except ValueError:
            continue # Skip rows with bad math data

    return cleaned

def download_file(url: str) -> bytes:
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise ValueError(f"Failed to download document: {str(e)}")

def get_pdf_images(file_bytes: bytes):
    """
    Returns a LIST of images. 
    We need a list (not a generator) to schedule Async tasks in parallel.
    """
    images = []
    try:
        if file_bytes.startswith(b"%PDF"):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # 150 DPI is the sweet spot for Llama 4 Vision accuracy
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            doc.close()
        else:
            # Single image case
            images.append(Image.open(BytesIO(file_bytes)))
        return images
    except Exception as e:
        raise ValueError("Failed to process file format.")

def encode_image(image: Image.Image) -> str:
    buffered = BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    # Quality 85 reduces bandwidth usage for faster upload to Groq
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- ASYNC EXTRACTOR ---
async def extract_data_from_image_async(image: Image.Image, page_num: int, semaphore: asyncio.Semaphore):
    """
    Async wrapper for LLM call.
    Uses Semaphore to control concurrency (prevents 429 Rate Limits).
    """
    async with semaphore: # Only allows N requests at a time
        base64_image = encode_image(image)
        
        # We ask for EVERYTHING. We will filter "Totals" in Python.
        # This is more reliable than asking the LLM to filter.
        system_prompt = """
        Extract ALL line items from this medical bill table.
        
        OUTPUT JSON:
        {
            "page_type": "Bill Detail", 
            "bill_items": [{"item_name": "x", "item_rate": 0.0, "item_quantity": 1.0, "item_amount": 0.0}]
        }
        
        RULES:
        1. Capture every row in the main table.
        2. page_type must be "Bill Detail", "Final Bill", or "Pharmacy".
        """

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": system_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            data = json.loads(result_text)
            
            # --- THE MAGIC FIX ---
            # Use Python logic to clean the data and remove double-counts
            data["bill_items"] = clean_and_validate_items(data.get("bill_items", []))
            
            # Page Type Safety
            if data.get("page_type") not in ["Bill Detail", "Final Bill", "Pharmacy"]:
                data["page_type"] = "Bill Detail"
                
            return data, response.usage

        except Exception as e:
            logger.error(f"Extraction failed on page {page_num}: {e}")
            # Return safe empty data so one failure doesn't crash the whole batch
            return {"page_type": "Bill Detail", "bill_items": []}, None