FROM python:3.10-slim

RUN pip install --no-cache-dir ray[default] scikit-learn flask

WORKDIR /app
COPY . .


