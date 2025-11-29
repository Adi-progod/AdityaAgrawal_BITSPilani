# Bill Extraction API

This project is a Vision-Language Model (VLM) based API designed to extract line items from complex medical bills. It utilizes Llama-4 Vision (via Groq) to parse PDFs and images, avoiding double-counting and adhering to a strict schema.

## Tech Stack
* **Framework:** FastAPI
* **AI Model:** Llama-4 Scout (Vision)
* **Language:** Python 3.10
* **Infrastructure:** Docker

## API Endpoint
`POST /extract-bill-data`

## Setup
1. Clone repository.
2. `pip install -r requirements.txt`
3. Install `poppler-utils`.
4. Set `GROQ_API_KEY` in `.env`.
5. Run `uvicorn main:app --reload`.