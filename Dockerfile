FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN gdown --id 1-nX-7krEGFB4YvCqO4Vyp8El367NecHV -O best_deeplabv3_model.pth

EXPOSE 5001

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5001

# Futáskor indítjuk a Flask alkalmazást
CMD ["flask", "run"]
