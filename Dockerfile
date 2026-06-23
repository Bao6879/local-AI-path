FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY inferenceStream.py .
COPY server.py .
COPY ui.html .
COPY checkpoints/ckpt.pt checkpoints/ckpt.pt

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]