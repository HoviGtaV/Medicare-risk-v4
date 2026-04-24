FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN mkdir -p /app/models_v4/final /app/examples

COPY score_batch_v4.py /app/score_batch_v4.py
COPY models_v4/final/model.cbm /app/models_v4/final/model.cbm

CMD ["python", "score_batch_v4.py", "--help"]
