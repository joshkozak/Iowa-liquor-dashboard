FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for geopandas
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 8080
EXPOSE 8080

# Run the application
CMD streamlit run --server.port 8080 --server.address 0.0.0.0 liquor_dashboard.py