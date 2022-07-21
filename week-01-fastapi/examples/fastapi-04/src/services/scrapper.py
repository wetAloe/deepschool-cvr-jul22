import http

import requests


class Scrapper:
    def __init__(self, config: dict):
        print('create scrapper')
        self.session = requests.session()
        self.url = config.url

    def get_desc_by_title(self, title):
        r = self.session.get(f'{self.url}/{title}')
        result = None
        if r.status_code == http.HTTPStatus.OK:
            result = r.text
        return result
