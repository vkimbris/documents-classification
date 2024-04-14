FROM python:3.10

WORKDIR /app

COPY requirements.txt ./requirements.txt


RUN pip3 install -r requirements.txt

EXPOSE 8002

COPY . /app


CMD streamlit run --server.port 8002 --server.enableCORS false --server.enableXsrfProtection false app.py