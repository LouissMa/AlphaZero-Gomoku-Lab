FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY alphazero_gomoku ./alphazero_gomoku
COPY models ./models
RUN python -m pip install --no-cache-dir ".[web]" \
    && useradd --create-home --uid 10001 gomoku \
    && chown -R gomoku:gomoku /app

USER gomoku
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)" || exit 1
CMD ["gomoku", "serve", "--host", "0.0.0.0", "--port", "8000"]