FROM python:3.11-slim

# WeasyPrint native deps for real PDF rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# CLI: mount a folder, convert a markdown report.
#   docker run --rm -v "$PWD":/work md-to-pdf /work/report.md -o /work/report.pdf
ENTRYPOINT ["python", "main.py"]
CMD ["--demo"]
