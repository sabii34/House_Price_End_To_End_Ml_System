FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY src ./src
COPY configs ./configs

# Create required dirs + train model inside image
RUN mkdir -p models data reports && \
    python -m src.data_loading && \
    python -m src.split && \
    python -m src.train

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "60"]
