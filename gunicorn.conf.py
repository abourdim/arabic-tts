# Gunicorn production config for MSA Arabic TTS
import multiprocessing, os

bind = f"{os.getenv('API_HOST','0.0.0.0')}:{os.getenv('API_PORT','8000')}"
workers = int(os.getenv('API_WORKERS', 1))
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
accesslog = "logs/access.log"
errorlog  = "logs/error.log"
loglevel  = os.getenv('LOG_LEVEL', 'info')
capture_output = True
