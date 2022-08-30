## Сегментация

[Слайды](https://docs.google.com/presentation/d/1DUOnTXlcjGM3Wji9jPiLMShqX6AHK46KwKYSCfJP8rQ/edit?usp=sharing)

### Задача

Решение задачи семантической сегментации на примере [конкурса с kaggle](https://www.kaggle.com/c/severstal-steel-defect-detection)
по детекции дефектов стали.

### Данные

Датасет весит _1.58 GB_. Скачать можно [отсюда](https://www.kaggle.com/c/severstal-steel-defect-detection/data).
Распаковываем в папку _data_ (можно поменять путь до папки в src/constants.py - DATA_PATH)


### Подготовка пайплайна

1. Создание и активация окружения
   ```bash
   python3 -m venv /path/to/new/virtual/environment
   source /path/to/new/virtual/environment/bin/activate
   ```

2. Установка пакетов

    В активированном окружении:
    ```bash
    pip install -r requirements.txt
    ```

3. Настройка Wandb
    - Создаем профиль [тут](https://wandb.ai/site)
    - Создаем в профиле новый проект (**Create new project**), куда
   будут сохраняться логи
    - Создаем ключ, чтобы подключаться к пользователю:
      - Тыкаем на иконку своего профиля в правом верхнем углу
      - Выбираем **Settings**
      - Ищем раздел **API keys**
      - Нажимаем **New key**
    - Добавляем в свое окружение ключ, который создали на предыдущем шаге:

   В активированном окружении
   ```bash
   wandb login <your-api-key>
   ```

4. В `src/config.py` меняем:
   - `project_name` - на имя проекта в wandb
   - `experiment_name` - на название своего эксперимента

   Остальное по усмотрению.

### Обучение

Чтобы запустить тренировку пишем:

```bash
nohup python -m run_train > nohup.out
```
