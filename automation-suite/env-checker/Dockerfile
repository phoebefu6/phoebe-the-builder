FROM python:3.11-slim

WORKDIR /app

COPY . .

# Stdlib-only checker. Mount your schema + .env and gate your deploy:
#   docker run --rm -v "$PWD":/work env-checker --schema /work/.env.schema --env-file /work/.env
ENTRYPOINT ["python", "main.py"]
CMD ["--demo"]
