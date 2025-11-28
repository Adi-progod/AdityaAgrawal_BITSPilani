from fastapi import FastAPI, HTTPException
from models import ExtractionRequest, ExtractionResponse, ExtractionData, PageLevelData, TokenUsage
from services import download_file, convert_to_images, extract_data_from_image
import logging

# Initialize App
app = FastAPI(title="HackRx Bill Extractor")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

@app.post("/extract-bill-data", response_model=ExtractionResponse)
async def extract_bill_data(request: ExtractionRequest):
    """
    Main endpoint to process bill documents (PDF/Image) and extract line items.
    """
    # Initialize counters
    total_tokens = 0
    in_tokens = 0
    out_tokens = 0
    extracted_pages = []
    global_item_count = 0

    try:
        logger.info(f"Processing document: {request.document}")
        
        # 1. Download
        file_bytes = download_file(request.document)
        
        # 2. Convert to Images (Handles PDFs and direct Images)
        images = convert_to_images(file_bytes)
        logger.info(f"Document converted to {len(images)} images.")

        # 3. Process Page by Page
        for idx, img in enumerate(images):
            page_num = idx + 1
            logger.info(f"Extracting page {page_num}...")
            
            # Call LLM Service
            data, usage = extract_data_from_image(img, page_num)
            
            # Handle Usage Stats
            if usage:
                in_tokens += usage.prompt_tokens
                out_tokens += usage.completion_tokens
                total_tokens += usage.total_tokens

            # Validate and Map to Pydantic
            # We ensure defaults are set if LLM misses a field
            clean_items = []
            for item in data.get("bill_items", []):
                clean_items.append({
                    "item_name": item.get("item_name", "Unknown Item"),
                    "item_amount": float(item.get("item_amount", 0.0)),
                    "item_rate": float(item.get("item_rate", 0.0)),
                    "item_quantity": float(item.get("item_quantity", 1.0))
                })

            page_data = PageLevelData(
                page_no=str(page_num),
                page_type=data.get("page_type", "Bill Detail"),
                bill_items=clean_items
            )
            
            extracted_pages.append(page_data)
            global_item_count += len(clean_items)

        # 4. Construct Success Response 
        return ExtractionResponse(
            is_success=True,
            token_usage=TokenUsage(
                total_tokens=total_tokens,
                input_tokens=in_tokens,
                output_tokens=out_tokens
            ),
            data=ExtractionData(
                pagewise_line_items=extracted_pages,
                total_item_count=global_item_count
            )
        )

    except Exception as e:
        logger.error(f"Critical Error: {str(e)}")
        # Return strict failure schema
        return ExtractionResponse(
            is_success=False,
            token_usage=TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0),
            data=ExtractionData(pagewise_line_items=[], total_item_count=0),
            error=str(e)
        )

# Health Check (Optional but good for deployment)
@app.get("/")
def read_root():
    return {"status": "Service is running", "model": "Llama 3.2 Vision (Groq)"}