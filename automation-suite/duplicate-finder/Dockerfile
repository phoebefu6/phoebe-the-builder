FROM python:3.11-slim

WORKDIR /app

COPY . .

# Stdlib-only CLI. Mount a folder and scan it:
#   docker run --rm -v "$PWD":/data duplicate-finder /data
#   docker run --rm -v "$PWD":/data duplicate-finder /data --delete
ENTRYPOINT ["python", "main.py"]
CMD ["--demo"]
