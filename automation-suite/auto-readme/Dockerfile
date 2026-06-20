FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Scan a mounted repo:
#   docker run --rm -v "$PWD":/repo auto-readme /repo -o /repo/README.md
ENTRYPOINT ["python", "main.py"]
CMD ["."]
