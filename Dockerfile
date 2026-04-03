FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r api/requirements.txt

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]