FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY railway_smoke.py /app/railway_smoke.py
EXPOSE 8080
CMD ["python", "/app/railway_smoke.py"]
