import base64
import requests
import os
import json
import logging
from io import BytesIO
from pdf2image import convert_from_bytes
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logger setup
logger = logging.getLogger("uvicorn")

# --- CONFIGURATION FOR GROQ (OPEN SOURCE) ---
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Updated to current Groq Vision model (Llama 4 Scout)
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

def download_file(url: str) -> bytes:
    """Downloads the file from the given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        raise ValueError(f"Failed to download document: {str(e)}")

def convert_to_images(file_bytes: bytes, content_type: str = "application/pdf") -> list:
    """Converts PDF bytes or Image bytes into a list of PIL Images."""
    try:
        # Simple heuristic: Check magic bytes or assume based on extension
        if file_bytes.startswith(b"%PDF"):
            return convert_from_bytes(file_bytes)
        else:
            return [Image.open(BytesIO(file_bytes))]
    except Exception as e:
        logger.error(f"Image conversion failed: {e}")
        raise ValueError("Failed to process file format. Ensure poppler-utils is installed.")

def encode_image(image: Image.Image) -> str:
    """Encodes PIL image to base64 string."""
    buffered = BytesIO()
    # Convert to RGB to handle PNGs with transparency
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_data_from_image(image: Image.Image, page_num: int):
    """
    Sends image to Groq Llama 3.2 Vision for extraction.
    """
    base64_image = encode_image(image)
    
    # SYSTEM PROMPT: The core logic to avoid double counting
    system_prompt = """
    You are an expert financial data extraction AI. Extract line items from this medical bill image.
    
    OUTPUT FORMAT (Strict JSON):
    {
        "page_type": "Bill Detail | Final Bill | Pharmacy",
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
    1. EXTRACT ONLY GENUINE LINE ITEMS (services, medicines, tests).
    2. IGNORE rows labeled: 'Total', 'Subtotal', 'Net Amount', 'Grand Total', 'Discount'.
    3. 'item_amount' must be the NET amount (Rate x Qty).
    4. If 'item_quantity' is missing, default to 1.0.
    5. If 'item_rate' is missing, infer from Amount/Qty.
    6. Return ONLY valid JSON.
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
        usage = response.usage
        
        # Parse JSON safely
        try:
            parsed_data = json.loads(result_text)
        except json.JSONDecodeError:
            # Fallback if model returns markdown ticks
            clean_text = result_text.replace("```json", "").replace("```", "")
            parsed_data = json.loads(clean_text)
            
        return parsed_data, usage

    except Exception as e:
        logger.error(f"LLM Extraction failed on page {page_num}: {e}")
        # Return empty safe structure on failure
        return {"page_type": "Bill Detail", "bill_items": []}, None