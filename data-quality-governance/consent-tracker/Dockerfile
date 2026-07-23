FROM python:3.11-slim

WORKDIR /app

# Install deps first so the layer caches when only app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Streamlit needs to bind 0.0.0.0 to be reachable from outside the container.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
