FROM python:3.11-slim

WORKDIR /app

# Install curl for uv installation if needed, or use pip
RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml README.md ./
COPY app/ ./app/

RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "-m", "app.cli"]

