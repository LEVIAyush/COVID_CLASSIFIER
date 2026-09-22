FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY flask_app.py .
COPY templates templates
# Either copy the model into the image...
COPY models models
# ...or leave models/ empty and set MODEL_URL at runtime instead.
ENV PORT=7860
EXPOSE 7860
CMD gunicorn -w 1 --threads 4 --timeout 120 -b 0.0.0.0:${PORT} flask_app:app
