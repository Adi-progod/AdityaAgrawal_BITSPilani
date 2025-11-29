import base64
import requests
import os
import json
import logging
import fitz  # PyMuPDF
import time  # <--- Added for rate limiting
from io import BytesIO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("uvicorn")

# --- CONFIGURATION ---
# Updated Client with explicit retry logic for Rate Limits
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    max_retries=5  # <--- Increased from default (2) to 5 to handle 429 errors better
)

# Using Llama 4 (Scout or Maverick)
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

def download_file(url: str) -> bytes:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        raise ValueError(f"Failed to download document: {str(e)}")

def process_pdf_pages(file_bytes: bytes):
    """
    Generator function that yields one PIL Image at a time from a PDF.
    This saves RAM by not loading all pages at once.
    """
    try:
        # Check if it's a PDF
        if file_bytes.startswith(b"%PDF"):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=150) # 150 DPI is enough for LLMs, saves RAM
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                yield img
                # Explicitly close page to free memory
                del img
                del pix
            doc.close()
        else:
            # Assume it's a single image file
            yield Image.open(BytesIO(file_bytes))
            
    except Exception as e:
        logger.error(f"File processing failed: {e}")
        raise ValueError("Failed to process file format.")

def encode_image(image: Image.Image) -> str:
    buffered = BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG", quality=85) # Quality 85 reduces base64 size
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_data_from_image(image: Image.Image, page_num: int):
    base64_image = encode_image(image)
    
    # UPDATED PROMPT: Removed the confusing " | " syntax from the JSON example
    system_prompt = """
    You are an expert financial data extraction AI. Extract line items from this medical bill image.
    
    OUTPUT FORMAT (Strict JSON):
    {
        "page_type": "Bill Detail", 
        "bill_items": [
            {
                "item_name": "string",
                "item_rate": float,
                "item_quantity": float,
                "item_amount": float
            }
        ]
    }
    
    CRITICAL RULES:
    1. "page_type" MUST be exactly one of these three strings: "Bill Detail", "Final Bill", or "Pharmacy".
    2. EXTRACT ONLY GENUINE LINE ITEMS.
    3. IGNORE rows labeled: 'Total', 'Subtotal', 'Grand Total'.
    4. If 'item_quantity' is missing, default to 1.0.
    5. Return ONLY valid JSON.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            temperature=0.1,
            max_tokens=4096,
            stream=False,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        data = json.loads(result_text)
        
        # --- SAFETY NET ---
        # Ensure page_type is valid before returning, otherwise Pydantic will crash.
        valid_types = ["Bill Detail", "Final Bill", "Pharmacy"]
        if data.get("page_type") not in valid_types:
            # If LLM hallucinates, default to "Bill Detail"
            data["page_type"] = "Bill Detail"

        # --- RATE LIMIT PROTECTION ---
        # Add a polite delay to allow the Token Bucket to refill
        time.sleep(2.0) 
            
        return data, response.usage

    except Exception as e:
        logger.error(f"LLM Extraction failed on page {page_num}: {e}")
        return {"page_type": "Bill Detail", "bill_items": []}, None