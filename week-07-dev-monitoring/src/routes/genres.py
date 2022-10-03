import cv2
import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest

from src.containers.conainers import AppContainer
from src.metrics.collectors import api_metrics_registry
from src.routes.routers import metrics_router
from src.routes.routers import poster_router
from src.services.poster_analysys import PosterAnalytics


@poster_router.get('/genres')
@inject
def genres_list(service: PosterAnalytics = Depends(Provide[AppContainer.poster_analytics])):
    return {
        'genres': service.genres,
    }


@poster_router.post('/predict')
@inject
def predict(
    image: bytes = File(),
    service: PosterAnalytics = Depends(Provide[AppContainer.poster_analytics]),
):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    genres = service.predict(img)
    return {'genres': genres}


@poster_router.post('/predict_proba')
@inject
def predict_proba(
    image: bytes,
    service: PosterAnalytics = Depends(Provide[AppContainer.poster_analytics]),
):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    return service.predict_proba(img)


@metrics_router.get('/metrics', response_class=PlainTextResponse)
def metrics():
    return generate_latest(api_metrics_registry)
