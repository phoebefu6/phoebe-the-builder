FROM python:3.11-slim

# node is not optional: the UTF-16 and ICU truncators are real JavaScript,
# not a Python imitation of it. A full-icu node is required for
# Intl.Segmenter to segment non-Latin scripts.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m pytest -q

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
