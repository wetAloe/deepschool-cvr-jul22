### Настройка окружения

Сначала создать и активировать venv:

```bash
python3 -m venv venv
. venv/bin/activate
```

Дальше поставить зависимости:

```bash
make install
```
### Команды

* `make install` - установка библиотек
* `make download_weights` - скачать веса моделек
* `make run_app` - запустить сервис


### Ссылка на материалы
* [DI контейнеры](https://python-dependency-injector.ets-labs.org/introduction/di_in_python.html)
* [Простой разбор, что такое Makefile](https://highload.today/make-i-makefile/)
* [FastApi](https://fastapi.tiangolo.com/)
* [Чистая архитектура АПИ сервисов (Clean API Architecture)](https://medium.com/perry-street-software-engineering/clean-api-architecture-2b57074084d5)
* [Крутая библиотека для работы с yaml конфигами (OmegaConf)](https://omegaconf.readthedocs.io/en/2.2_branch/)
