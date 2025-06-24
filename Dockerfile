FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .


RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y procps net-tools curl gnupg lsb-release && apt-get clean
RUN pip install --no-cache-dir -r requirements.txt
RUN curl -fsSL https://get.docker.com | sh

COPY . .


EXPOSE 8000 8265 10001 6379

CMD [ "python", "api.py" ]