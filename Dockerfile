# Gunakan Python 3.9 slim
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file proyek
COPY . .

# Buat folder uploads
RUN mkdir -p uploads

# Beri izin eksekusi script download
RUN chmod +x download_model.sh

# Expose port (Railway akan handle ini)
EXPOSE $PORT

# Command untuk menjalankan app
CMD ["sh", "-c", "./download_model.sh && gunicorn --bind 0.0.0.0:$PORT api:app"]