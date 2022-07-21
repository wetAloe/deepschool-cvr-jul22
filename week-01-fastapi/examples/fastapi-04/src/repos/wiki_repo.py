import os


class WikiRepo:
    """
    Класс для сохранения и получения объектов
    """
    def __init__(self, config: dict):
        print('create repo')
        self.dir = config['dir']

    def save(self, filename, text):
        with open(f'{self.dir}/{filename}', 'w') as f:
            f.write(text)

    def get(self, filename):
        if filename not in os.listdir(self.dir):
            return None

        with open(f'{self.dir}/{filename}', 'r') as f:
            result = f.read()
        return result
