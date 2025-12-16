FROM python:3.10-slim

# Install dependencies
WORKDIR /root/ml-app
COPY requirements.txt requirements.txt

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends 
RUN apt-get install ffmpeg -y
RUN rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt