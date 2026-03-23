FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU-only first (avoids pulling the CUDA build)
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    flask==3.0.0 \
    numpy==1.26.4 \
    pandas==2.2.0 \
    scikit-learn==1.4.0

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
