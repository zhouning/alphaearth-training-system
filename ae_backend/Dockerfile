FROM python:3.11-slim

WORKDIR /app

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system dependencies for spatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
