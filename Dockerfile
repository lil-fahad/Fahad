FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/tmp/hf
WORKDIR /app
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
RUN python -m pip install --no-cache-dir "transformers>=5.16,<6" "accelerate>=1.14,<2" "einops>=0.8,<1" huggingface-hub safetensors pyyaml packaging psutil httpx numpy requests && \
    python -m pip install --no-cache-dir --no-deps chronos-forecasting==2.3.1
COPY chronos_benchmark.py /app/chronos_benchmark.py
CMD ["python", "/app/chronos_benchmark.py"]
