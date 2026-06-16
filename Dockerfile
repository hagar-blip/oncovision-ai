FROM python:3.11

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxcb1

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD uvicorn app:app --host 0.0.0.0 --port $PORT