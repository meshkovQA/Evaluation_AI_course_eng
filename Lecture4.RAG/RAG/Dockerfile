FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash app

WORKDIR /home/app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY .env.example .env

# Create storage directories with correct permissions
RUN mkdir -p storage/documents storage/vector_db/chroma && \
    chown -R app:app /home/app

# Switch to application user
USER app

EXPOSE 8021

ENV PYTHONPATH=/home/app
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8021"]
