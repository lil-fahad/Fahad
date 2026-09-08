FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY runtime.*.b64 /tmp/runtime-parts/
RUN python - <<'PY'
import base64, pathlib, zipfile
parts=sorted(pathlib.Path('/tmp/runtime-parts').glob('runtime.*.b64'))
raw=base64.b64decode(''.join(p.read_text() for p in parts))
pathlib.Path('/tmp/runtime.zip').write_bytes(raw)
with zipfile.ZipFile('/tmp/runtime.zip') as z: z.extractall('/app')
PY
RUN pip install --no-cache-dir .
RUN mkdir -p /data/state /data/models && chmod 700 /data /data/state /data/models
EXPOSE 8080
CMD ["python", "-m", "telegram_bridge.railway_bootstrap"]
