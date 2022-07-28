import cv2
import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File

from src.containers.conainers import AppContainer
from src.routes.routers import router
from src.services.poster_analysys import PosterAnalytics


@router.get('/genres')
@inject
def genres_list(service: PosterAnalytics = Depends(Provide[AppContainer.poster_analytics])):
    return {
        'genres': service.genres,
    }


@router.post('/predict')
@inject
def predict(
    image: bytes = File(),
    service: PosterAnalytics = Depends(Provide[AppContainer.poster_analytics]),
):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    genres = service.predict(img)

    return {'genres': genres}


@router.post('/predict_proba')
@inject
def predict_proba(
    image: bytes = File(),
    service: PosterAnalytics = Depends(Provide[AppContainer.poster_analytics]),
):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    return service.predict_proba(img)


@router.get('/health_check')
def health_check():
    return 'OK'
