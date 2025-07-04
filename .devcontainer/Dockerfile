FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get install -y git