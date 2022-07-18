## Лекция 1. Моделлинг

Решение задачи мультилейбл классификации на примере определения жанра фильма по постеру.

[Ссылка](https://docs.google.com/presentation/d/1BIJbfKYNZsnCyRl2SqMWvW-wsHTCa-2K9r7D9w0vf_E/edit?usp=sharing) на презентацию к лекции 1.

### Датасет

Включает 7254 постеров и 25 жанров.
Скачать датасет и данные можно [отсюда](https://drive.google.com/file/d/1xiNgtQoOmwEZDkmCGvB7EsezU3kgtLpq/view?usp=sharing).

### Подготовка пайплайна

1. Создание и активация окружения
    ```
    python3 -m venv /path/to/new/virtual/environment
    ```
    ```
    source /path/to/new/virtual/environment/bin/activate
    ```

2. Установка пакетов

    В активированном окружении:
    ```
    pip install -r requirements.txt
    ```

3. Разбиваем датасет на 3 части:
    ```
    ROOT_PATH=/data/movie-genre-classification python train_test_split.py
    ```

4. Настройка ClearML
    - [в своем профиле ClearML](https://app.community.clear.ml/profile) нажимаем "Create new credentials"
    - у себя на компе пишем `clearml-init` и следуем инструкциям

### Обучение
Запуск тренировки c `nohup`:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/movie-genre-classification nohup python train.py > log.out
```

Запуск тренировки без `nohup`:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/movie-genre-classification python train.py
```

### Добавление модели в DVC
1. Инициализация DVC

    В директории проекта пишем команды:
    ```
    dvc init
   ```
    ```
    dvc remote add --default myremote gdrive://gdrive_folder_id
    ```

    где gdrive_folder_id берется из адреса вашей папки в Google Drive https://drive.google.com/drive/folders/gdrive_folder_id

    ```
    dvc config cache.type hardlink,symlink
    dvc checkout --relink
   ```

    подробнее про типы линков можно почитать [здесь](https://dvc.org/doc/user-guide/large-dataset-optimization#file-link-types-for-the-dvc-cache).

2. Добавление модели в DVC
    
    Копируем в `weights` обученную модель
    ```
    cd weights
    dvc add model.pt
    dvc push
   ```

3. Делаем коммит с новой моделью:
    ```
    git add .
    git commit -m "add new model"
   ```

### Загрузка модели из DVC к себе
   ```
    git pull origin master
    dvc pull
   ```

### [History](HISTORY.md) экспериментов

### Немного про ClearML

В рамках курса мы рассматриваем ClearML
как удобный логгер, который позволяет удобно шарить и хранить
эксперименты внутри команды. На самом деле, почти все команды, 
которые я знаю, только так его и используют.

Но ClearML это не только логгер, у него есть много других клёвых
возможностей
* Делать контроллеры пайплайнов и подбирать гиперпараметры. [Тык](https://clear.ml/docs/latest/docs/guides/pipeline/pipeline_controller) и [тык](https://clear.ml/docs/latest/docs/guides/optimization/hyper-parameter-optimization/examples_hyperparam_opt);
* [ClearML AGENT](https://github.com/allegroai/clearml-agent). Запуск экспериментов на удаленной машине (или в облаке), можно прямо из UI, подкручивая нужные параметры;
* [Управлять данными](https://github.com/allegroai/clearml/blob/master/docs/datasets.md);
* Не так давно у них даже появился [свой сервинг](https://github.com/allegroai/clearml-serving).

Ещё у них есть [ютуб-канал](https://www.youtube.com/c/ClearML/featured), где они коротко и по делу рассказывают о своих
возможностях.

И нужно немного позанудствовать. ~~охужэтотвендорлок~~. Перед тем, как полностью пересаживать
все свои процессы на ClearML, вспомните о том, что это стартап. И о том, что может
случаться со стартапами.  Это не попытка отговорить, просто учитывайте 
этот риск.