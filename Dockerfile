# Imagen base con Python
FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY app/ ./app/
COPY scripts/ ./scripts/

# Crear directorios necesarios
RUN mkdir -p data models logs

# Variables de entorno
ENV PYTHONPATH=/app
ENV RAY_DISABLE_IMPORT_WARNING=1

# Puerto por defecto para FastAPI
EXPOSE 8000

# Comando por defecto (se sobrescribe en docker-compose)
CMD ["python", "app/main.py"]