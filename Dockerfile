# Python 3.12 matches both the environment that produced the published results
# and the hosted Streamlit build, so all three run the same interpreter series.
FROM python:3.12-slim

# Created before the dependency layer so that editing the application does not
# invalidate it.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Dependencies first. Docker caches this layer, so a change to app.py rebuilds
# in seconds rather than reinstalling numpy, pandas and scipy every time.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY risk.py app.py ./

# Nothing in this image needs root.
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/_stcore/health')"

# 0.0.0.0 rather than localhost: inside a container localhost means the
# container itself, and the port would be unreachable from outside it.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
