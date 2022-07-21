import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from src.containers.containers import AppContainer
from src.routes.routers import wiki_router
from src.routes import wiki as wiki_routes


def create_app() -> FastAPI:
    # Инициализация DI контейнера
    container = AppContainer()
    # Инициализация конфига
    cfg = OmegaConf.load('config/config.yml')
    # Прокидываем конфиг в наш контейнер
    container.config.from_dict(cfg)
    # Говорим контейнеру, в каких модулях он будет внедряться
    container.wire([wiki_routes])

    app = FastAPI()
    # цепляем роутер к нашему приложению
    app.include_router(wiki_router, prefix='/wiki', tags=['wiki'])
    return app


app = create_app()
if __name__ == '__main__':
    uvicorn.run(app, port=1234, host='0.0.0.0')
