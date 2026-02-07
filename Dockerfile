FROM python:3.10-slim

# 1) system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2) workdir
WORKDIR /app

# 3) install python deps first (cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) copy code + config + model artifact
COPY app ./app
COPY src ./src
COPY configs ./configs
COPY models ./models

# 5) expose port
EXPOSE 8000

# 6) run in production style:
# gunicorn manages workers, uvicorn runs ASGI
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "60"]
