# Use an official lightweight Python image
FROM python:3.10-slim

# 1. Install system dependencies (Poppler is required for pdf2image)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Copy dependencies first (for caching)
COPY requirements.txt .

# 4. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
COPY . .

# 6. Expose the port (Render uses port 10000 by default, but we'll bind dynamically)
EXPOSE 8000

# 7. Command to run the application
# We use the PORT environment variable provided by the cloud host
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]