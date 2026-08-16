@echo off
set TONGYI_API_KEY=sk-22c2c40f50594c22b6e5ed06b9681e72
set VLLM_BASE_URL=http://localhost:8101/v1
set OTEL_SDK_DISABLED=true
echo Starting LLM Gateway...
E:\workspace\shop-agent\venv\Scripts\python.exe -m uvicorn gateway.main:app --host 0.0.0.0 --port 8001
echo Gateway exited with code %ERRORLEVEL%