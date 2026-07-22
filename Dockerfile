FROM python:3.11-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ ./src/
COPY model_export/ ./model_export/

EXPOSE 8001

CMD ["uvicorn", "serve_model:app", "--host", "0.0.0.0", "--port", "8001", "--app-dir", "src"]