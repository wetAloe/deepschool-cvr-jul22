from time import sleep

import requests
import uvicorn
import sentry_sdk

from fastapi import FastAPI
from omegaconf import DictConfig
from omegaconf import OmegaConf
from sentry_sdk.integrations.fastapi import FastApiIntegration
from starlette.middleware.base import BaseHTTPMiddleware

from src.containers.conainers import AppContainer
from src.middleware import prometheus_middleware
from src.routes.routers import poster_router as poster_router, metrics_router
from src.routes import genres as poster_routes

def init_sentry(cfg: DictConfig):
    sentry_sdk.init(
        dsn=cfg['dsn'],
        integrations=[FastApiIntegration()]
    )

def create_app() -> FastAPI:
    container = AppContainer()
    cfg = OmegaConf.load('config/config.yml')
    container.config.from_dict(cfg)
    container.wire([poster_routes])
    init_sentry(cfg.sentry)
    app = FastAPI()
    app.add_middleware(BaseHTTPMiddleware, dispatch=prometheus_middleware)
    set_routers(app)
    return app


def set_routers(app: FastAPI):
    app.include_router(poster_router, prefix='/poster', tags=['poster'])
    app.include_router(metrics_router, tags=['metrics'])


app = create_app()

def ddos(url: str):
    for _ in range(1000000):
        r = requests.get(url)
        print(r.text)
        sleep(0.2)

if __name__ == '__main__':
    uvicorn.run(app, port=2444, host='0.0.0.0')
