FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# CLI tool: pass a CSV path as the container argument.
#   docker run --rm -v "$PWD":/data csv-cleaner /data/messy.csv
ENTRYPOINT ["python", "main.py"]
CMD ["--demo"]
