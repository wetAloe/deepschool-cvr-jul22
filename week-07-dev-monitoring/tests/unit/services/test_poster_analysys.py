from copy import deepcopy

import cv2
import numpy as np

from src.containers.conainers import AppContainer


class FakeRotateClassifier:

    def predict(self, image):
        return cv2.ROTATE_90_CLOCKWISE


def test_predicts_not_fail(app_container: AppContainer, sample_image_np: np.ndarray):
    poster_analytics = app_container.poster_analytics()
    poster_analytics.predict(sample_image_np)
    poster_analytics.predict_proba(sample_image_np)


def test_prob_less_or_equal_to_one(app_container: AppContainer, sample_image_np: np.ndarray):
    poster_analytics = app_container.poster_analytics()
    genre2prob = poster_analytics.predict_proba(sample_image_np)
    for prob in genre2prob.values():
        assert prob <= 1


def test_predict_dont_mutate_initial_image(app_container: AppContainer, sample_image_np: np.ndarray):
    initial_image = deepcopy(sample_image_np)
    poster_analytics = app_container.poster_analytics()
    poster_analytics.predict(sample_image_np)

    assert np.allclose(initial_image, sample_image_np)


def test_rotated_image_dont_mutate_initial_image(app_container: AppContainer, sample_image_np: np.ndarray):
    initial_image = deepcopy(sample_image_np)

    with app_container.reset_singletons():
        with app_container.rotate_classifier.override(FakeRotateClassifier()):
            poster_analytics = app_container.poster_analytics()

    poster_analytics.predict(sample_image_np)

    assert np.allclose(initial_image, sample_image_np)
