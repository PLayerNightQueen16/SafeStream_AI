FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# 🔥 Mode switch using environment variable
CMD ["sh", "-c", "if [ \"$MODE\" = \"eval\" ]; then python inference.py; else uvicorn server.app:app --host 0.0.0.0 --port 7860; fi"]