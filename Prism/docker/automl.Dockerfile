# Dockerfile for the AutoML backend service
FROM python:3.11-slim

WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose port and run the API
EXPOSE 8001
CMD ["uvicorn", "autoML.api:app", "--host", "0.0.0.0", "--port", "8001"]
