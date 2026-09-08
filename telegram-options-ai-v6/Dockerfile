FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY runtime.*.b64 /tmp/runtime-parts/
RUN python - <<'PY'
import base64, hashlib, pathlib, zipfile
parts=sorted(pathlib.Path('/tmp/runtime-parts').glob('runtime.*.b64'))
assert len(parts) == 8, f'expected 8 runtime parts, got {len(parts)}'
raw=base64.b64decode(''.join(p.read_text() for p in parts))
actual=hashlib.sha256(raw).hexdigest()
expected='f3ef6475d6a699da870be36cd992b2c87f8b6f666a0b26a2846b7d8a8c569344'
assert actual == expected, f'runtime sha256 mismatch: {actual}'
pathlib.Path('/tmp/runtime.zip').write_bytes(raw)
with zipfile.ZipFile('/tmp/runtime.zip') as z: z.extractall('/app')
print('runtime sha256 verified', actual)
PY
RUN pip install --no-cache-dir .
RUN mkdir -p /data/state /data/models && chmod 700 /data /data/state /data/models
EXPOSE 8080
CMD ["python", "-m", "telegram_bridge.railway_bootstrap"]
