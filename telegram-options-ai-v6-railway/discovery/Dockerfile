FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY discovery.py /app/discovery.py
EXPOSE 8080
CMD ["python", "/app/discovery.py"]
