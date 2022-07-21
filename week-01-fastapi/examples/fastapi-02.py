from typing import Optional
from typing import Union

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None


class ItemOut(BaseModel):
    item_name: str
    item_id: int


@app.get("/hello")
def read_root():
    """
    Простейшний ендпоинт, возращающий hello world
    :return:
    """
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    """
    :param item_id: path параметр
    :param q: query параметр
    :return:
    """
    return {"item_id": item_id, "q": q}


@app.post("/items/{item_id}")
def create_item(item_id: int, item: Item):
    """
    :param item_id: path параметр
    :param item: данные передающиеся в теле запроса в виде json
    :return:
    """
    return {"item_name": item.name, "item_id": item_id}


@app.post("/items_new/{item_id}", response_model=ItemOut)
def create_item_new(item_id: int, item: Item):
    """
    Тоже самое, что предыдущие, но с проставленным
    response_model (Теперь в документации есть пример возвращаемых данных)
    :param item_id:
    :param item:
    :return:
    """
    return ItemOut(**{"item_name": item.name, "item_id": item_id})


uvicorn.run(app, host='0.0.0.0', port=7070)