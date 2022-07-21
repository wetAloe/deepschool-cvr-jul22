from dependency_injector import containers, providers

from src.repos.wiki_repo import WikiRepo
from src.services.scrapper import Scrapper
from src.services.wiki_service import WikiService


class AppContainer(containers.DeclarativeContainer):
    """
    Класс DI контейнера
    """
    config = providers.Configuration()

    # зависимость wiki репозиторий
    wiki_repo = providers.Singleton(
        WikiRepo,
        config=config.repos.wiki_repo,
    )

    # зависимость парсер (scrapper)
    scrapper = providers.Factory(
        Scrapper,
        config=config.services.scrapper,
    )

    # зависимость сервиса с осной бизнес логикой
    wiki_service = providers.Factory(
        WikiService,
        wiki_repo=wiki_repo,
        scrapper=scrapper,
    )
