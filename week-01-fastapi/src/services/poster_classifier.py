import typing as tp

import numpy as np
import torch
import torch.nn as nn

from src.services.preprocess_utils import preprocess_image


class PosterClassifier:

    def __init__(self, config: tp.Dict):
        self._model_path = config['model_path']
        self._device = config['device']

        self._model: nn.Module = torch.jit.load(self._model_path, map_location=self._device)
        self._classes: np.ndarray = np.array(self._model.classes)
        self._size: tp.Tuple[int, int] = self._model.size
        self._thresholds: np.ndarray = np.array(self._model.thresholds)

    @property
    def classes(self) -> tp.List:
        return list(self._classes)

    @property
    def size(self) -> tp.Tuple:
        return self._size

    def predict(self, image: np.ndarray) -> tp.List[str]:
        """Предсказание списка жанров.

        :param image: RGB изображение;
        :return: список жанров.
        """
        return self._postprocess_predict(self._predict(image))

    def predict_proba(self, image: np.ndarray) -> tp.Dict[str, float]:
        """Предсказание вероятностей принадлежности к жанрам.

        :param image: RGB изображение.
        :return: словарь вида `жанр фильма`: вероятность.
        """
        return self._postprocess_predict_proba(self._predict(image))

    def _predict(self, image: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей.

        :param image: RGB изображение;
        :return: вероятности после прогона модели.
        """
        batch = preprocess_image(image, self._size)

        with torch.no_grad():
            model_predict = self._model(batch.to(self._device)).detach().cpu()[0]

        return model_predict.numpy()

    def _postprocess_predict(self, predict: np.ndarray) -> tp.List[str]:
        """Постобработка для получения списка жанров.

        :param predict: вероятности после прогона модели;
        :return: список жанров.
        """
        return self._classes[predict > self._thresholds].tolist()

    def _postprocess_predict_proba(self, predict: np.ndarray) -> tp.Dict[str, float]:
        """Постобработка для получения словаря с вероятностями.

        :param predict: вероятности после прогона модели;
        :return: словарь вида `жанр фильма`: вероятность.
        """
        return {self._classes[i]: float(predict[i]) for i in predict.argsort()[::-1]}
