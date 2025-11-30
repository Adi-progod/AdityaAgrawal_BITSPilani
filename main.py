from fastapi import FastAPI
from models import ExtractionRequest, ExtractionResponse, ExtractionData, PageLevelData, TokenUsage
from services import download_file, get_pdf_images, extract_data_from_image_async
import logging
import asyncio
import uvicorn
import time

app = FastAPI(title="HackRx Bill Extractor")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

@app.post("/extract-bill-data", response_model=ExtractionResponse)
async def extract_bill_data(request: ExtractionRequest):
    start_time = time.time()
    total_tokens = 0
    in_tokens = 0
    out_tokens = 0
    extracted_pages = []
    global_item_count = 0

    try:
        logger.info(f"Processing document...")
        file_bytes = download_file(request.document)
        
        # 1. Load Images (Fast CPU operation)
        images = get_pdf_images(file_bytes)
        logger.info(f"Loaded {len(images)} pages. Starting Parallel Extraction...")

        # 2. CONCURRENCY CONTROL (The Speed Tuner)
        # Semaphore(3) means "Process 3 pages at exactly the same time".
        # This keeps us under Groq's rate limit while being 3x faster than sequential.
        sem = asyncio.Semaphore(3) 

        # 3. Create Async Tasks
        tasks = []
        for idx, img in enumerate(images):
            page_num = idx + 1
            # We schedule the task but don't await it yet
            tasks.append(extract_data_from_image_async(img, page_num, sem))

        # 4. Fire Parallel Requests! 
        # This waits for all batches to finish.
        results = await asyncio.gather(*tasks)

        # 5. Process Results (They return in the correct order 1, 2, 3...)
        for idx, (data, usage) in enumerate(results):
            page_num = idx + 1
            
            if usage:
                in_tokens += getattr(usage, 'prompt_tokens', 0)
                out_tokens += getattr(usage, 'completion_tokens', 0)
                total_tokens += getattr(usage, 'total_tokens', 0)

            # Map the clean data to our Schema
            clean_items = []
            # Note: The data["bill_items"] here is already cleaned by services.py logic
            for item in data.get("bill_items", []):
                clean_items.append({
                    "item_name": str(item.get("item_name", "Unknown")),
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

        duration = time.time() - start_time
        logger.info(f"✅ Completed {len(images)} pages in {duration:.2f} seconds.")

        return ExtractionResponse(
            is_success=True,
            token_usage=TokenUsage(total_tokens=total_tokens, input_tokens=in_tokens, output_tokens=out_tokens),
            data=ExtractionData(pagewise_line_items=extracted_pages, total_item_count=global_item_count)
        )

    except Exception as e:
        logger.error(f"Critical Error: {str(e)}")
        return ExtractionResponse(
            is_success=False,
            token_usage=TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0),
            data=ExtractionData(pagewise_line_items=[], total_item_count=0),
            error=str(e)
        )

if __name__ == "__main__":
    from pyngrok import ngrok
    try:
        # Auto-connect ngrok for convenience
        ngrok.kill()
        public_url = ngrok.connect(8000).public_url
        print(f"\n🚀 PUBLIC API URL: {public_url}/extract-bill-data\n")
    except Exception as e:
        print(f"Ngrok error: {e}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)