import pytest

from example_tests.examples import Exchanger

@pytest.fixture
def default_rate():
    return 10

@pytest.fixture
def exchanger(default_rate):
    return Exchanger(rate=default_rate)
