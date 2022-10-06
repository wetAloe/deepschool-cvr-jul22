import json
from functools import lru_cache

import aiohttp
from fastapi import FastAPI, Depends, File
from omegaconf import OmegaConf
from pydantic import BaseSettings

from constants import CONFIG_PATH

app = FastAPI()


class Settings(BaseSettings):
    served_model_name: str
    predictions_endpoint: str


@lru_cache
def get_app_settings():
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    return Settings(
        served_model_name=cfg['served_model']['model_name'],
        predictions_endpoint=cfg['ts']['predictions_endpoint'],
    )


@app.post('/predict_proba')
async def predict_proba(
    image: bytes = File(),
    settings: Settings = Depends(get_app_settings),
):
    ######################################
    # some business logic here if need it#
    ######################################
    predict_url = f'{settings.predictions_endpoint}/predictions/{settings.served_model_name}'
    async with aiohttp.ClientSession() as session:
        async with session.post(predict_url, data={'data': image}) as resp:
            res = await resp.read()
    ######################################
    # some business logic here if need it#
    ######################################
    return json.loads(res)
