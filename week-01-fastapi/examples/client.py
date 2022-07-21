import json

import requests

port = 7070
BASE_URL = f'http://localhost:{port}'

# fastapi-01
r = requests.get(f'{BASE_URL}/hello')
print(type(r.text), type(r.json()))
print(json.loads(r.text))
print(type(json.loads(r.text)))

# fastapi-02
r = requests.get(f'{BASE_URL}/items/1')
print(r.json())

r = requests.get(f'{BASE_URL}/items/1?q=2')
print(r.json())

r = requests.get(f'{BASE_URL}/items/1', params=dict(q=3))
print(r.url)
print(r.json())

body = {
    'name': 'foobar',
    'price': 1.54
}
#
r = requests.post(f'{BASE_URL}/items/3', data=body)
print(r.json())

r = requests.post(f'{BASE_URL}/items/3', data=json.dumps(body))
print(r.json())
#
r = requests.post(f'{BASE_URL}/items/3', json=body)
print(r.json())

# fastapi-03-bad/ fastapi-04
body = dict(name='алфавит')
r = requests.post(f'{BASE_URL}/wiki', json=body)
print(r.json())

body = dict(name='алфавит')
r = requests.get(f'{BASE_URL}/wiki/алфавит', json=body)
print(r.json())

# requests.session
session = requests.session()
print(session.headers)
print(session.cookies)

session.get('https://google.com')
print('*' * 100)
print(session.headers)
print(session.cookies)

session.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, ' \
                                'like Gecko) Chrome/102.0.0.0 Safari/537.36 '
session.get('https://yandex.ru')
print('*' * 100)
print(session.headers)
print(session.cookies)
#
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/102.0.0.0 Safari/537.36',
           'Accept-Encoding': 'gzip, deflate, br', 'Accept': '*/*', 'Connection': 'keep-alive'}
requests.get('https://google.com', headers=headers, cookies=session.cookies)
requests.get('https://yandex.ru', headers=headers, cookies=session.cookies)
