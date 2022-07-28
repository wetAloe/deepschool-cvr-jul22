from fastapi.testclient import TestClient
from http import HTTPStatus


def test_genres_list(client: TestClient):
    response = client.get('/poster/genres')
    assert response.status_code == HTTPStatus.OK

    genres = response.json()['genres']

    assert isinstance(genres, list)


def test_predict(client: TestClient, sample_image_bytes: bytes):
    files = {
        'image': sample_image_bytes,
    }
    response = client.post('/poster/predict', files=files)

    assert response.status_code == HTTPStatus.OK

    predicted_genres = response.json()['genres']

    assert isinstance(predicted_genres, list)


def test_predict_proba(client: TestClient, sample_image_bytes: bytes):
    files = {
        'image': sample_image_bytes,
    }
    response = client.post('/poster/predict_proba', files=files)

    assert response.status_code == HTTPStatus.OK

    genre2prob = response.json()

    for genre_prob in genre2prob.values():
        assert genre_prob <= 1
