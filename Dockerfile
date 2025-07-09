FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todos los archivos
COPY . .

# Exponer puertos
EXPOSE 8265 10001 5000 8080

CMD [ "python", "api.py" ]