from dependency_injector import containers, providers

from src.services.poster_analysys import PosterAnalytics
from src.services.poster_classifier import PosterClassifier
from src.services.rotate_classifier import RotateClassifier


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    poster_classifier = providers.Factory(
        PosterClassifier,
        config=config.services.poster_classifier,
    )

    rotate_classifier = providers.Singleton(
        RotateClassifier,
        config=config.services.rotate_classifier,
    )

    poster_analytics = providers.Singleton(
        PosterAnalytics,
        rotate_classifier=rotate_classifier,
        poster_classifier=poster_classifier,
    )
