import pytest

from example_tests.examples import Exchanger
from example_tests.examples import sum_func


def test_sum():
    assert sum_func(1, 0) == 1
    assert sum_func(5, 3) == 8
    assert sum_func(-1, 3) == 55

def test_exchanger_work(exchanger: Exchanger):
    exchanger.exchange(10)

def test_exchanger_on_positive(exchanger: Exchanger, default_rate: int):
    assert exchanger.exchange(10) == default_rate * 10

@pytest.mark.skip(reason='to do')
def test_exchanger_on_negative(exchanger: Exchanger):
    assert exchanger.exchange(-1) == 0
