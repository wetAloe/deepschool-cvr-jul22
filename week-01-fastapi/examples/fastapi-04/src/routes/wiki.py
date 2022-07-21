import http

from dependency_injector.wiring import Provide, inject
from fastapi import Depends
from fastapi import HTTPException
from pydantic import BaseModel

from src.containers.containers import AppContainer
from src.routes.routers import wiki_router
from src.services.wiki_service import WikiService


class WikiName(BaseModel):
    name: str


class WikiOut(BaseModel):
    name: str
    description: str


@wiki_router.post('')
@inject
def create_wiki(wiki_name: WikiName, wiki_service: WikiService = Depends(Provide[AppContainer.wiki_service])):
    result = wiki_service.process_description(wiki_name.name)
    if result is None:
        raise HTTPException(
            status_code=http.HTTPStatus.NOT_FOUND,
            detail=f'Page {wiki_name.name} not found'
        )
    return 'OK'


@wiki_router.get('/{name}', response_model=WikiOut)
@inject
def get_wiki(name: str, wiki_service: WikiService = Depends(Provide[AppContainer.wiki_service])):
    description = wiki_service.get_description(name)
    if description is None:
        raise HTTPException(
            status_code=http.HTTPStatus.NOT_FOUND,
            detail=f'File {name} not found'
        )
    return WikiOut(
        name=name,
        description=description,
    )
