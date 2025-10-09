# --- Базовый образ и рабочая директория
FROM python:3.11-slim
WORKDIR /app

# --- Установка системных зависимостей (если нужны сборки)
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# --- Копирование и установка Python-зависимостей
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

# --- Копирование кода
COPY . .

# --- Экспорт PATH для запуска из виртуального окружения
ENV PATH="/opt/venv/bin:$PATH"

# --- Порт приложения (измените при необходимости)
EXPOSE 8000

# --- Команда запуска: замените на вашу (Flask/FastAPI/Django)
# Примеры:
#   Gunicorn для WSGI (Flask/Django)
#   CMD ["gunicorn", "your_module:app", "--bind", "0.0.0.0:8000", "--workers", "2"]
#   Uvicorn для ASGI (FastAPI)
#   CMD ["uvicorn", "your_module:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "2"]
