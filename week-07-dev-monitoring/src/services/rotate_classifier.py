from typing import Dict, List, Tuple

import numpy as np
import torch

from src.services.preprocess_utils import preprocess_image

NOT_ROTATE_CODE = 3


class RotateClassifier:

    def __init__(self, config: Dict):
        self._model_path = config['model_path']
        self._device = config['device']
        self._model: torch.nn.Module = torch.jit.load(self._model_path, map_location=self._device)
        self._size: Tuple[int, int] = self._model.size
        self._rotate_codes: np.ndarray = np.array(self._model.rotate_codes)

    def predict(self, image: np.ndarray) -> List[str]:
        """Предсказание как нужно повернуть изображение.

        :param image: RGB изображение;
        :return: коды для поворотов: 0-2 - код для функции cv2.rotate, 3 - поворот не нужен.
        """
        return self._postprocess(self._predict(image))

    def _predict(self, image: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей.

        :param image: RGB изображение;
        :return: вероятности после прогона модели.
        """
        batch = preprocess_image(image, self._size).to(self._device)

        with torch.no_grad():
            model_predict = self._model(batch).detach().cpu()[0]

        return model_predict.cpu().numpy()

    def _postprocess(self, predict: np.ndarray) -> List[str]:
        """Постобработка для получения кода поворота.

        :param predict: вероятности после прогона модели;
        :return: код поворота.
        """
        return self._rotate_codes[predict.argmax()]
