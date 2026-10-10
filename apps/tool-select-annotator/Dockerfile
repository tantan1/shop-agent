FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend

EXPOSE 8137 8501

# 默认启动后端 API;Streamlit UI 由 compose 中以同镜像不同 command 启动
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8137"]
