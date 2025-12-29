FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY app .

ENV PORT 8501
ENV HOST 0.0.0.0

CMD [ "sh", "-c", "streamlit run --server.port ${PORT} --server.address ${HOST} /app/dashboard.py" ]