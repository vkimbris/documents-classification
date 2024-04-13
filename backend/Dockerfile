FROM python:3.10-slim

RUN apt-get update -y && apt-get install -y g++

WORKDIR /backend

COPY . /backend

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["bash", "run.sh"]