FROM python:3.9-slim

WORKDIR /app

# Install system dependencies needed for some Python packages (like LightGBM)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and data
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "freightfox_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
