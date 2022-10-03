from copy import deepcopy

import numpy as np

from src.containers.conainers import AppContainer


def test_predicts_not_fail(app_container: AppContainer, sample_image_np: np.ndarray):
    rotate_classifier = app_container.rotate_classifier()
    rotate_classifier.predict(sample_image_np)


def test_predict_dont_mutate_initial_image(app_container: AppContainer, sample_image_np: np.ndarray):
    initial_image = deepcopy(sample_image_np)
    rotate_classifier = app_container.rotate_classifier()
    rotate_classifier.predict(sample_image_np)

    assert np.allclose(initial_image, sample_image_np)
