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

#### Подготовка
* `make install` - установка библиотек
* `make download_weights` - скачать веса моделек

#### Запуск сервиса
* `make run_app` - запустить сервис. Можно с аргументом `APP_PORT`

#### Сборка образа
* `make build` - собрать образ. Можно с аргументами `DOCKER_TAG`, `DOCKER_IMAGE`

#### Статический анализ
* `make lint` - запуск линтеров

#### Тестирование
* `make run_unit_tests` - запуск юнит-тестов
* `make run_integration_tests` - запуск интеграционных тестов
* `make run_all_tests` - запуск всех тестов
* `make generate_coverage_report` - сгенерировать coverage-отчет

### Ссылка на материалы
* [Презентация](https://docs.google.com/presentation/d/1Ra6mupBY27zfoK9njn7g7jTiN0E8TMFPFUR6BAi1Ixg/edit?usp=sharing)
* [Интерактивный и холиварный доклад про линтеры от wemake](https://www.youtube.com/watch?v=7IVCOzL41Lk&ab_channel=PythonChannel)
* [Продвинутое использование pytest](https://www.youtube.com/watch?v=7KgihdKTWY4&t=675s&ab_channel=%D0%92%D0%B8%D0%B4%D0%B5%D0%BE%D1%81%D0%BA%D0%BE%D0%BD%D1%84%D0%B5%D1%80%D0%B5%D0%BD%D1%86%D0%B8%D0%B9IT-People)
* [TDD: когда нужно и, самое главное, когда не нужно](https://www.youtube.com/watch?v=QT1yDL-L0t4&ab_channel=HighLoadChannel)
* [Основы контейнеризации ](https://habr.com/ru/post/659049/)
* [pre-commit](https://pre-commit.com/)
