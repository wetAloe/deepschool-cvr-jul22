# DI-контейнеры

## Выделяем зависимости

### Простой пример из жизни
Начнем рассказывать о внедрении зависимостей с простого примера.

Пусть нам нужно предсказать картинку и записать предсказание в базу данных.

```python

class DataBase:
    def __init__(self, connect_params):
        """Например, создаем соединение"""
        self._connection = some_db_lib.connect(**connect_params)

    def save(self, predict):
        """Сохраняем предикт"""


class Model:
    def __init__(self, checkpoint_path):
        """Подгружаем чекпоинт, ставим на нужный девайс и т.п."""

    def predict(self, image):
        """Предсказываем картинку"""

        return predict


class ImageHandler:
    def __init__(self):
        self._model = Model('checkpoint.pt')
        self._db = DataBase(
            {
                'db_url': 'http://my.db.org',
                'user': 'my-name',
                'password': 'my-secret-password',
            },
        )

    def predict(self, image):
        prediction = self._model.predict(image)
        self._db.save(prettified_predict)
        return prediction

```

Что не так с этим кодом?

Мы захотели добавить тест на класс `ImageHandler`. Например,
тест будет проверять, что с предиктом все окей. Что-то типа такого

```python
def test_image_handler():
    image = get_some_image()
    ok_predict = get_ok_predict_for_these_image()
    handler = ImageHandler()

    predict = handler.predict(image)

    assert predict == ok_predict
```

Ничего не смущает?

При **каждом** прогоне теста мы будем что-то сохранять в боевую базу. Это точно не дела!

Как от этого избавиться? Сейчас наша боевая база прибита гвоздями в ините у `ImageHandler`.
Сохранение в БД из метода `predict` тоже не убрать, потому что тогда
при реальной эксплуатации нашего `ImageHandler`'а ничего сохраняться не будет, а нужно чтоб сохранялось.

Было бы хорошо, чтобы в продакшне использовалась настоящая база данных, а в тестах
какая-нибудь фиктивная, типа такой:

```python
class FakeDB:
    def __init__(self, connection_params):
        self._db = []

    def save(self, predict):
        self._db.append(predict)
```

Просто списочек в памяти, в который не грех записать что-нибудь при тесте.

Как бы нам этого достичь?

Выделим зависимости!

`````{tab-set}
````{tab-item} Было

```{code-block} python

class ImageHandler:
    def __init__(self):
        self._model = Model('checkpoint.pt')
        self._db = DataBase(
            {
                'db_url': 'http://my.db.org',
                'user': 'my-name',
                'password': 'my-secret-password',
            },
        )
```
````

````{tab-item} Стало

```{code-block} python
class ImageHandler:
    def __init__(self, model, db):
        self._model = model
        self._db = db
```
````
`````

Мы заменили прибитые гвоздями модель и базу на зависимость `ImageHandler`'а от модели и базы.
Теперь наш хендлер опирается только на то, что те объекты, которые к нему пришли реализуют тот интерфейс,
который он от них ждет. А именно:

* у `model` есть метод `.predict(image)`
* у `db` есть метод `.save(predict)`

Теперь мы сможем создать разный `ImageHandler` под разные сценарии:

`````{tab-set}
````{tab-item} Тест

```{code-block} python
model = Model('checkpoint.pt')
db = FakeDB({})
handler = ImageHandler(model, db)
```
````

````{tab-item} Прод

```{code-block} python
model = Model('checkpoint.pt')
db = DataBase(
    {'db_url': 'http://my.db.org', 'user': 'my-name', 'password': 'my-password'},
)
handler = ImageHandler(model, db)
```
````
`````


Что нам даёт такой простой трюк:
* **Гибкость**. Сегодня мы хотим записывать в Postgres. А завтра могут
поменяться требования и мы решим записывать в ClickHouse. Раньше нам нужно было бы
лезть в код `ImageHandler` и руками что-то менять. А сейчас мы просто передадим ему другой объект, у которого
есть метод `.save(predict)`. Сам код хендлера меняться не будет
* **Просто тестировать**. Думаю, пример выше это хорошо показал:) 

Вообще, если ваш код тяжело тестировать, то это сильный признак того, что с ним что-то не так.
Нужно бежать рефакторить.

```{seealso}
Мы сделали наш код лучше с точки зрения дизайн-правила *Low Coupling and High Cohesion*. [Подробнее про него](https://medium.com/german-gorelkin/low-coupling-high-cohesion-d36369fb1be9)
```

Модельку для тестов можно тоже делать ненастоящей, если от нас этого не требует
конкретный тест. Можно сделать, например, такую:

```python
class StubModel:
    def __init__(self, checkpoint_path):
        """Здесь ничего не происходит"""

    def predict(self, image):
        return "cat"
```
Можно заметить, что для `FakeDB` и `StubModel` мы сделали сигнатуру метода `__init__` такой же
как у продакшн-классов. Это нужно для того, чтобы наши фейковые классы полностью совпадали по интерфейсу
с нефейковыми.

Посмотрим еще на один приятный бонус. Пусть мы хотим написать тест на то,
что `ImageHandler` пытается записывать в базу. Лезть в прод-базу в тестах конечно же не нужно.
Воспользуемся нашей подделкой:

```python
def test_image_handler_save_predict():
    image = get_some_image()
    model = StubModel('lalala')
    db = FakeDB({})
    handler = ImageHandler(model, db)
    
    #  лезть в приватные поля классов плохо даже в тестах,
    #  но чтобы не раздувать гайд, идем на компромисс
    initial_db_size = len(db._db)
    
    handler.predict(image)
    
    db_size_after_predict = len(db._db)
    
    assert db_size_after_predict == initial_db_size + 1
```

Мы протестировали, что `ImageHandler` что-то сохраняет и нам для этого не
понадобились ни настоящая модель ни настоящая база. Красота!

### Проблемы с тем, чтобы эти зависимости оказались где нужно

```{note}
Если вы читаете этот гайд до лекции по FastAPI, то какие-то вещи, специфичные
для FastAPI могут быть непонятны. Тогда выдыхаем и не концентрируемся на них во время прочтения 
гайда. После гайда просто нужно будет еще раз вернуться к ним:) 
```

На курсе мы рассматриваем [FastAPI](https://fastapi.tiangolo.com), но все что будет написано ниже
будет верно для любого фреймворка. 

В FastAPI нам привычна такая конструкция для функций-контроллеров:

```python
@router.post('/predict')
def predict(image: bytes):
    # где-то 
    # здесь 
    # предсказываем
    return prediction
```

Но чтобы предсказать, нам нужна наша модель и, возможно, какая-нибудь база, в которую мы что-нибудь запишем.
Нужно, чтобы внутри функции у нас был доступ до модели. Как этого добиться?

Первый очевидный выход - заведем глобальную переменную:

```python
model = Model('checkpoint.pt')
db = DataBase(
    {'db_url': 'http://my.db.org', 'user': 'my-name', 'password': 'my-password'},
)

handler = ImageHandler(model, db)


@router.post('/predict')
def predict(image: bytes = File()):
    # какая-то 
    # возможная 
    # предобработка
    
    return handler.predict(image)
```

Это даже будет работать. Но мы опять столкнемся с проблемой, что
в функции `predict(image)` прибит гвоздями `handler`. Мы опять сильно потеряли в гибкости, за которую бились.

Есть еще один вариант:

```python
from functools import partial


model = Model('checkpoint.pt')
db = DataBase(
    {'db_url': 'http://my.db.org', 'user': 'my-name', 'password': 'my-password'},
)

handler = ImageHandler(model, db)

def predict(image: bytes, handler: ImageHandler):
    # какая-то 
    # возможная 
    # предобработка
    
    return handler.predict(image)

predict = partial(predict, handler=handler)
predict = router.post('/predict')(predict)

```

Теперь в функции `predict` `handler` стал зависимостью. И сама функция стала гибче.
Но согласитесь, вся мишура вокруг выглядит не очень красиво. А теперь представьте, что в реальных
приложениях у нас могут быть десятки зависимостей и сотни функций/классов, куда бы мы хотели их занести подобным образом.
Выглядит как ад.

Здесь нам на помощь приходят библиотеки и фреймворки, которые помогают нам описывать и внедрять зависимости (*Dependency Injection*)
очень просто и удобно.

## Python-Dependency-Injector

У многих фреймворков есть свой механизм внедрения зависимостей. Например, можно посмотреть на
механизмы в [FastAPI](https://fastapi.tiangolo.com/tutorial/dependencies/) или в [BlackSheep](https://github.com/Neoteroi/BlackSheep#automatic-bindings-and-dependency-injection).

На курсе будем использовать библиотеку [dependency-injector-python](https://python-dependency-injector.ets-labs.org/introduction/di_in_python.html).

Она не привязана ни к какому фреймворку и ее можно легко использовать как в FastAPI, Flask, Django, ...,  так и просто в своем безфреймворковом коде. 

Начнем разбираться прям с примера из их документации:

Пусть у нас есть `ApiClient` клиент какого-то API. Чтобы собраться, ему нужен ключик для API и таймаут.
У нас так же есть `Service`, в котором есть какая-то бизнес-логика над `ApiClient`. Мы уже
И есть функция `main(service)`, которая как-то эксплуатирует наш `Service`.

````python
import os


class ApiClient:

    def __init__(self, api_key: str, timeout: int) -> None:
        self.api_key = api_key
        self.timeout = timeout


class Service:

    def __init__(self, api_client: ApiClient) -> None:
        self.api_client = api_client


def main(service: Service) -> None:
    ...
````

Заметим, что мы уже выделили все зависимости и не хардкодим их посреди кода. Наши классы
довольно гибкие. Но чтобы это собралось, нам нужно написать такую матрешку:

```python
if __name__ == "__main__":
    main(
        service=Service(
            api_client=ApiClient(
                api_key=os.getenv('API_KEY'),
                timeout=int(os.getenv('TIMEOUT')),
            ),
        ),
    )
```

Таких матрешек по нашему приложению может насобираться очень много. И может стать не понятно, где и как, кто и от кого собирается.
### Кто такой контейнер

Чтобы описать, какой класс из каких классов собирается, в этой библиотеке используется класс `DeclarativeContainer` (в дальнейшем
будем звать его просто контейнером).
```python
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject


class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

    api_client = providers.Singleton(
        ApiClient,
        api_key=config.api_key,
        timeout=config.timeout,
    )

    service = providers.Factory(
        Service,
        api_client=api_client,
    )
```

Пока не вдаемся в новые слова и пытаемся просто прочесть, что написано
в этом куске кода:
* Контейнер - это просто класс, в котором записаны объекты, которыми мы хотим в 
будущем жонглировать
* Слева написано, какой объект
* Справа - из чего он собирается

Например, мы можем прочесть, что `api_client` - это что-то связанное
с классом `ApiClient`, которое соберется при помощи ключевых аргументов
`api_key` и `timeout`.

`service` - это что-то связанное с классом `Service` и оно соберется при помощи ключевого
аргумента `api_client`.

```{note}
В правиле для сборки `service` мы используем `api_client=api_client`. То есть
это именно тот клиент, который мы парой строчек выше описали как собирать.
```

Мощь этого подхода можно раскрыть на чуть более усложненном примере:
пусть у нас есть `ApiClient1`, `ApiClient2`. Есть `Service1` - класс, который
зависит только от `ApiClient1`. `Service2` - зависит только от `ApiClient2`. `Service3` - 
зависит от `ApiClient1` и `ApiClient2`.

Соорудим контейнер для такой ситуации:

```python
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject


class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

    api_client1 = providers.Singleton(
        ApiClient1,
        api_key=config.api_key1,
        timeout=config.timeout1,
    )
    
    api_client2 = providers.Singleton(
        ApiClient2,
        api_key=config.api_key2,
        timeout=config.timeout2,
    )

    service1 = providers.Factory(
        Service1,
        api_client=api_client1,
    )
    
    service2 = providers.Factory(
        Service2,
        api_client=api_client2,
    )

    service3 = providers.Factory(
        Service3,
        api_client1=api_client1,
        api_client2=api_client2,
    )
```

Мы всего один раз описали, как собирать `api_client1`, 
но смогли в двух местах (`service1` и `service3`) описать правила сборки при помощи него.

### Кто такие providers

Разберемся, кто такие эти `providers`.

Провайдеры помогают собирать объекты. Их [довольно много](https://python-dependency-injector.ets-labs.org/providers/index.html)
. Здесь рассмотрим `Factory` и `Singleton`.

Начнем с `Factory`. Прочтем этот код:

```python
from dependency_injector.providers import Factory


class ApiClient:
    def __init__(self, token, timeout):
        self._token = token
        self._timeout = timeout

    def get_token(self):
        return self._token

    def get_timeout(self):
        return self._timeout


if __name__ == '__main__':
    token = '123'
    timeout = 345

    client_provider = Factory(ApiClient, token=token, timeout=timeout)
    client = client_provider()

    assert isinstance(client, ApiClient)

    assert client.get_timeout() == timeout
    assert client.get_token() == token
```

Что нужно вынести:

`Factory` это провайдер, которому мы говорим, каким образом собирать объект.
Первым аргументом мы передаем класс, который нужно заинитить. Следующими аргументами - аргументы, которыми
мы будем инициализировать этот класс.

Чтобы затем получить сам объект класса `AppClient`, мы должны позвать
метод `__call__` у провайдера, т.е. сделать `client_provider()`.

Посмотрим на еще один важный кусок кода:
```python
from dependency_injector.providers import Factory


class ApiClient:
    def __init__(self, token, timeout):
        self._token = token
        self._timeout = timeout

    def get_token(self):
        return self._token

    def get_timeout(self):
        return self._timeout


if __name__ == '__main__':
    token = '123'
    timeout = 345

    client_provider = Factory(ApiClient, token=token, timeout=timeout)
    client1 = client_provider()
    client2 = client_provider()

    assert id(client1) != id(client2)

    assert isinstance(client1, ApiClient)
    assert isinstance(client2, ApiClient)
```

То есть мы два раза позвали `client_provider()` и получили **два разных** объекта
класа `ApiClient`.

А теперь посмотрим на `Singleton`:
```python
from dependency_injector.providers import Factory, Singleton


class ApiClient:
    def __init__(self, token, timeout):
        self._token = token
        self._timeout = timeout

    def get_token(self):
        return self._token

    def get_timeout(self):
        return self._timeout


if __name__ == '__main__':
    token = '123'
    timeout = 345

    client_provider = Singleton(ApiClient, token=token, timeout=timeout)
    client1 = client_provider()
    client2 = client_provider()
    assert id(client1) == id(client2)

    assert isinstance(client1, ApiClient)
```

Т.е. `Singleton` это такой провайдер, который сколько бы раз мы не дёрнули, он будет возвращать **один и тот же объект**.
В этом главное различие `Singleton` и `Factory`.

````{admonition} Закрепим про провайдеры
:class: important
Провайдеры - это такие callable классы, которые методом `__call__` соберут нам объект
с нужными аргументами.
```python
#  это провайдер
client_provider = Factory(ApiClient, token='123', timeout=1) 

# а если его дёрнуть, то получим объект нужного класса
client = client_provider()  
```
````

### Магия, ради которой этот гайд

Вспомним матрешку, которую мы писали до этого:

```python
import os


class ApiClient:

    def __init__(self, api_key: str, timeout: int) -> None:
        self.api_key = api_key
        self.timeout = timeout


class Service:

    def __init__(self, api_client: ApiClient) -> None:
        self.api_client = api_client


def main(service: Service) -> None:
    ...

if __name__ == "__main__":
    main(
        service=Service(
            api_client=ApiClient(
                api_key=os.getenv('API_KEY'),
                timeout=int(os.getenv('TIMEOUT')),
            ),
        ),
    )
```

До этого нам не нравилось, что таких матрешек по нашему приложению может насобираться очень много. И может стать не 
понятно, где и как, кто и от кого собирается.

Напишем такой код и чуть ниже попытаемся его осознать.

```python
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject


class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

    api_client = providers.Singleton(
        ApiClient,
        api_key=config.api_key,
        timeout=config.timeout,
    )

    service = providers.Factory(
        Service,
        api_client=api_client,
    )


@inject
def main(service: Service = Provide[Container.service]) -> None:
    ...


if __name__ == "__main__":
    container = Container()
    container.config.api_key.from_env('API_KEY', required=True)
    container.config.timeout.from_env('TIMEOUT', as_=int, default=5)
    container.wire(modules=[__name__])

    main()
```

Что сейчас произошло. Все зависимости мы не собирали около вызова нужной нам функции.
Мы декларативно описали их в контейнере.

В этом коде мы говорим, чтобы контейнер сам зашел во все функции, которые мы пометили декоратором
`@inject` и внедрил в них зависимости. 
Т.е. сейчас мы зовём функцию `main()` без аргументов, потому что 
контейнер зашел в нее и сделал так, чтобы дефолтное значение аргумента
`service` стало равно объекту класса `Service`. Этот объект мы описали как собирать в самом
контейнере.

Какие новые слова в этом коде:
* `@inject` помечаем функцию как функцию, в которую должен зайти контейнер и внедрить зависимости
* `service=Provide[Container.service]`
   * `Provide` - это **маркер** помечает, что ключевой аргумент `service` контейнер должен переопределить
   * `Provide[Container.service]` говорит контейнеру, что ключевой аргумент `service` нужно заменить на
  то, что вернет провайдер `Container.service`. Т.е. после того, как контейнер потрогает этот ключевой аргумент,
  у функции `main` в качестве дефолтного значение для аргумента `service` вместо `Provide[Container.service]` станет
  `Container.service()`. Т.е. объект класса `Service`, который мы описали как собирать в самом контейнере
* `container.wire(modules=[__name__])`: говорим контейнеру, в какие модули нужно зайти, чтобы внедрить зависимости.

```{note}
Мы описали все зависимости нашего приложения в `Container`. Затем мы хотим,
чтобы все эти зависимости внедрились в нужные функции. Для этого мы помечаем нужные
функции декоратором `@inject`, а дефолтные значения аргументов для внедрения маркером  `Provide`.
После вызова `container.wire(modules=modules_list)` дефолтные значения в маркированных
аргументах помеченных функций заменятся на результат вызова соответствующего провайдера.
И саму функцию мы сможем позвать просто как `main()` без аргументов, потому что все нужные ей
аргументы уже зашились в дефолтные значения.

```

``` {seealso}
Для любителей копнуть поглубже сделал [борду](https://miro.com/app/board/uXjVPcwYDsc=/?share_link_id=1972549274). В ней 
рассказано, как вся эта магия устроена под капотом.
```

### Как это всё подружить с FastAPI

Вспомним изначальный пример с FastAPI, который нам не понравился:

```python
model = Model('checkpoint.pt')
db = DataBase(
    {'db_url': 'http://my.db.org', 'user': 'my-name', 'password': 'my-password'},
)

handler = ImageHandler(model, db)


@router.post('/predict')
def predict(image: bytes = File()):
    # какая-то 
    # возможная 
    # предобработка
    
    return handler.predict(image)
```

Теперь перепишем это в нашем DI-стиле. Код в карусельке нужно воспринимать
как отдельные питонячьи модули.

`````{tab-set}
````{tab-item} routes
```{code-block} python
from dependency_injector.wiring import Provide, inject

from conainers import Container
from routers import router
from services import ImageHandler

@router.post('/predict')
@inject
def predict(
    image: bytes = File(),
    handler: ImageHandler = Depends(Provide[Container.image_handler]),
):
    return handler.predict(image)

```
````

````{tab-item} services
```{code-block} python
"""
Здесь просто классы с нужной логикой
"""

class DataBase:
    def __init__(self, connect_params):
        """Например, создаем соединение"""
        self._connection = some_db_lib.connect(**connect_params)

    def save(self, predict):
        """Сохраняем предикт"""


class Model:
    def __init__(self, checkpoint_path):
        """Подгружаем чекпоинт, ставим на нужный девайс и т.п."""

    def predict(self, image):
        """Предсказываем картинку"""
        # где-то здесь
        return predict
        
        
class ImageHandler:
    def __init__(self, model, db):
        self._model = model
        self._db = db
        
    def predict(self, image):
        prediction = self._model.predict(image)
        self._db.save(prettified_predict)
        return prediction
```
````
````{tab-item} containers
``` {code-block} python
from dependency_injector import containers, providers

from services import DataBase, Model, ImageHandler



class Container(containers.DeclarativeContainer):
    """Здесь описываем, кто от кого зависит"""
    config = providers.Configuration()

    model = providers.Singleton(
        Model,
        checkpoint_path=config.checkpoint_path,
    )

    db = providers.Factory(
        DataBase,
        connect_params=config.connect_params,
    )

    image_handler = providers.Factory(
        ImageHandler,
        model=model,
        db=db,
    )

```
````
````{tab-item} routers
```{code-block} python
from fastapi import APIRouter

router = APIRouter()

```
````
````{tab-item} main
```{code-block} python
"""
А здесь мы собираем приложение и внедряем зависимости
"""


import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

import routes
from conainers import Container
from routers import router 



def create_app() -> FastAPI:
    container = Container()
    #  в этом конфиге лежит все, что нам нужно: путь до чекпоинта и 
    #  креды для базы
    cfg = OmegaConf.load('config.yml') 
    container.config.from_dict(cfg)
    container.wire([routes])

    app = FastAPI()
    app.include_router(router, prefix='/my-prefix', tags=['some_tag'])
    return app


app = create_app()

if __name__ == '__main__':
    uvicorn.run(app, port=1337, host='0.0.0.0')

```
````
`````

В модуле `routes` у нас появилась новая директива `Depends`. Если вкратце, она
нужна, чтобы наша библиотека дружила с FastAPI. Если подробнее - снова приглашаю в [борду](https://miro.com/app/board/uXjVPcwYDsc=/?share_link_id=1972549274) :)

На первый взгляд может показаться, что мы навернули сложностей на этот код. Но это верно только для такого коротенького
примерчика. Когда наш код разрастется до нескольких тысяч строчек, мы вспомним идею внедрения зависимостей с большой теплотой:)

### Про перегрузку зависимостей:

Пусть мы хотим написать тест, который тестирует наш `ImageHandler`. Мы помним,
что не хотим в тестах дёргать прод-базу. Посмотрим, как элегантно можно подменить
прод-базу на тестовую всего одним менеджером контекста:

`````{tab-set}
````{tab-item} test
```{code-block} python

from conainers import Container
from test_utils import FakeDB

def test_handler():
  image = get_some_image()
  correct_predict = get_correct_predict_for_these_image()
  container = Container()
  container.config.checkpoint_path = '/path/to/test/model.pt'
  
  with container.db.override(FakeDB({})):
      #  Помним, что внутри контейнера лежат провайдеры
      #  поэтому вызов image_handler() отдаст нам объект класса ImageHandler
      handler = container.image_handler()
      predict = handler.predict(image)
   
  assert predict == correct_predict

```
````
````{tab-item} services
```{code-block} python
"""
Здесь просто классы с нужной логикой
"""

class DataBase:
    def __init__(self, connect_params):
        """Например, создаем соединение"""
        self._connection = some_db_lib.connect(**connect_params)

    def save(self, predict):
        """Сохраняем предикт"""


class Model:
    def __init__(self, checkpoint_path):
        """Подгружаем чекпоинт, ставим на нужный девайс и т.п."""

    def predict(self, image):
        """Предсказываем картинку"""
        # где-то здесь
        return predict
        
        
class ImageHandler:
    def __init__(self, model, db):
        self._model = model
        self._db = db
        
    def predict(self, image):
        prediction = self._model.predict(image)
        self._db.save(prettified_predict)
        return prediction
```
````
````{tab-item} containers
``` {code-block} python
from dependency_injector import containers, providers

from services import DataBase, Model, ImageHandler



class Container(containers.DeclarativeContainer):
    """Здесь описываем, кто от кого зависит"""
    config = providers.Configuration()

    model = providers.Singleton(
        Model,
        checkpoint_path=config.checkpoint_path,
    )

    db = providers.Factory(
        DataBase,
        connect_params=config.connect_params,
    )

    image_handler = providers.Factory(
        ImageHandler,
        model=model,
        db=db,
    )

```
````
````{tab-item} test_utils
```{code-block} python
```python
class FakeDB:
    def __init__(self, connection_params):
        self._db = []

    def save(self, predict):
        self._db.append(predict)
```
```
````
   
`````

Красота! Всего в одном месте мы сказали `container.db.override(FakeDB({}))` и прод-база в дальнейшем 
коде (внутри менеджера) подменилась на фейковую! 

Если вас не восхищает такая простота, то представьте,
что для того чтобы собраться, вашему хендлеру нужно (с учетом того, что его аргументам тоже понадобятся зависимости)
скажем 20 зависимостей. Раньше вам бы пришлось руками писать матрешку из 20 зависимостей, чтобы
собрать тестовый `ImageHandler`. А сейчас мы всего одной строчкой подменили ровно то, что нам нужно!

