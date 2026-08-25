# The readers are real interpreters, so the image has to carry them.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        nodejs git gawk perl ruby jq sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# `python evidence.py` prints the full audit; the app is the interactive view.
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
