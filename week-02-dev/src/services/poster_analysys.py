from typing import Dict, List

import cv2
import numpy as np

from src.services.poster_classifier import PosterClassifier
from src.services.rotate_classifier import RotateClassifier, NOT_ROTATE_CODE


class PosterAnalytics:

    def __init__(self, rotate_classifier: RotateClassifier, poster_classifier: PosterClassifier):
        self._rotate_classifier = rotate_classifier
        self._poster_classifier = poster_classifier

    @property
    def genres(self):
        return self._poster_classifier.classes

    def predict(self, image: np.ndarray) -> List[str]:
        """Предсказания списка жанров по постеру.

        :param image: входное RGB изображение;
        :return: список жанров.
        """

        rotate_code = self._rotate_classifier.predict(image)

        image = self._rotate_image(image, rotate_code)

        return self._poster_classifier.predict(image)

    def predict_proba(self, image: np.ndarray) -> Dict[str, float]:
        """Предсказание вероятностей принадлежности постера к жанрам.

        :param image: входное RGB изображение;
        :return: словарь вида `жанр фильма`: вероятность.
        """

        rotate_code = self._rotate_classifier.predict(image)

        if rotate_code != NOT_ROTATE_CODE:
            image = cv2.rotate(image, rotate_code)

        return self._poster_classifier.predict_proba(image)

    def _rotate_image(self, image: np.ndarray, rotate_code: int) -> np.ndarray:
        if rotate_code != NOT_ROTATE_CODE:
            image = cv2.rotate(image, rotate_code)
        return image
