import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from src.containers.conainers import AppContainer
from src.routes.routers import router as app_router
from src.routes import genres as genre_routes


def create_app() -> FastAPI:
    container = AppContainer()
    cfg = OmegaConf.load('config/config.yml')
    container.config.from_dict(cfg)
    container.wire([genre_routes])

    app = FastAPI()
    app.include_router(app_router, prefix='/poster', tags=['poster'])
    return app


app = create_app()

if __name__ == '__main__':
    uvicorn.run(app, port=2444, host='0.0.0.0')