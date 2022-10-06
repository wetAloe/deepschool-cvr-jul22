import json
import logging
import typing as tp

import requests
from omegaconf import OmegaConf

from constants import CONFIG_PATH
from utils import init_logging


class TorchServeModelUpdater:
    def __init__(self, config: dict):
        self._management_endpoint = config['ts']['management_endpoint']

        served_model_cfg = config['served_model']
        self._request_params = served_model_cfg['request_params']
        self._model_name = served_model_cfg['model_name']

    def update(self):
        previous_versions = self._get_previous_versions()
        for version in previous_versions:
            self._remove_previous_version(version)
        self._upload_model()

    def _get_previous_versions(self) -> tp.List[str]:
        url = f'{self._management_endpoint}/models/{self._model_name}'
        response = requests.get(url)
        if response.ok:
            available_models = json.loads(response.text)
            return [model['modelVersion'] for model in available_models]
        return []

    def _remove_previous_version(self, version: str):
        url = f'{self._management_endpoint}/models/{self._model_name}'
        delete_params = {'model_version': version}
        response = requests.delete(url, params=delete_params)
        response.raise_for_status()
        logging.info(f'Model {self._model_name} with version {version} was deleted, Response {response.text}')

    def _upload_model(self):
        url = f'{self._management_endpoint}/models'
        response = requests.post(url, params=self._request_params)
        response.raise_for_status()
        logging.info(f'Model {self._model_name} was uploaded. Response {response.text}')


if __name__ == '__main__':
    init_logging()
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    model_updater = TorchServeModelUpdater(cfg)
    model_updater.update()
