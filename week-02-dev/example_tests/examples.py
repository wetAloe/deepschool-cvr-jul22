def sum_func(a: int, b: int) -> int:
    return a + b

class Exchanger:
    def __init__(self, rate: int):
        print('Exchanger created')
        self._rate = rate

    def exchange(self, money: int):
        return money * self._rate
