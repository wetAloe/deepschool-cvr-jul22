import time

from fastapi import Request

from src.metrics.collectors import requests_counter
from src.metrics.collectors import requests_latency_hist


async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    # Выполняем запрос
    response = await call_next(request)
    request_time = time.time() - start_time
    # Проставляем метрики только для router genres
    if request.url.path.startswith('/poster'):
        requests_counter.labels(request.method, request.url.path).inc()
        requests_latency_hist.labels(request.method, request.url.path).observe(request_time * 1000)
    return response
