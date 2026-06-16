FROM python:3.12-slim

LABEL maintainer="ollama-sim"
LABEL description="Faux serveur Ollama qui route vers Mistral API"

WORKDIR /app

# Dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY main.py .

# Volume pour les logs
RUN mkdir -p /app/logs
VOLUME ["/app/logs"]

# Port Ollama standard
EXPOSE 11434

# Variables d'environnement (à surcharger au runtime)
ENV MISTRAL_API_KEY=""
ENV MISTRAL_BASE_URL="https://api.mistral.ai/v1"

# Lancement
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "11434", "--log-level", "info"]
