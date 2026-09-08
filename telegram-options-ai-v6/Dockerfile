FROM python:3.12-slim AS tested
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY . .
RUN sha256sum --check runtime.sha256
RUN pip install --no-cache-dir -r requirements-dev.lock
RUN python -m pytest -q
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock
COPY --from=tested /wheels /tmp/wheels
RUN pip install --no-cache-dir --no-deps /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels /tmp/requirements.lock
RUN mkdir -p /data/state /data/models && chmod 700 /data /data/state /data/models
EXPOSE 8080
CMD ["python", "-m", "telegram_bridge.railway_bootstrap"]
