## Лекция 2. Моделлинг, мультилейбл классификация

Решение задачи мультилейбл классификации на примере определения жанра фильма по постеру.

[Ссылка](https://docs.google.com/presentation/d/1_d87WPo_WcQLf5fBsYi2kw0e3pzwzozOi-wceP1qhoI/edit?usp=sharing) на презентацию к лекции 2.

### Отличия от предыдущего пайплайна тренировки

1) Поддержка нескольких конфигов тренировки в репозитории. Полезно, если вы хотите сначала обучить голову
с замороженным бэкбоном, а потом файнтюнить всю сеть целиком. Теперь для запуска необходимо явно указать файл конфига:
```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/movie-genre-classification python train.py configs/simple_config.py
```

2) Скрипт для SWA. При запуске также нужно указать путь до конфиг файла из которого будет браться вся необходимая информация:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/movie-genre-classification python swa.py configs/simple_config.py
```
Полученные веса сохранятся в папку эксперимента с именем `model.swa.pth`.  
[Ссылка](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/) на "правильное" swa в PyTorch.

3) Добавилось несколько новых компонентов в файл конфигурации:
 - `callbacks` - список калбэков, относящиеся только к текущему эксперименту;
 - `num_iteration_on_epoch` - количество итераций на одну эпоху, работает через [семплер](src/dataset.py),
используется только для тренировочного лоадера;
 - `checkpoint_name` - имя претренированного чекпоинта.

### Пример пайплайна обучения

##### 1) Находим lr для обучения головы с замороженным бэкбоном.
Запускаем скрипт:
```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/movie-genre-classification python train.py configs/lrf_config_1.py
```
Не пугаемся, что в конце пайплайн падает с ошибкой, так задумано)  
После завершения скрипта идём в [тетрадку](notebooks/lrf.ipynb). Смотрим на график, определяемся с диапазоном.

##### 2) Обучения головы с замороженным бэкбоном.
После найденного диапазона lr запускаем скрипт с обучением.
```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/movie-genre-classification python train.py configs/train_only_head.py
```

##### 3) Пробуем найти lr для файнтюнинга.
Запускаем скрипт:
```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/movie-genre-classification python train.py configs/lrf_config_2.py
```
Идём в ту же [тетрадку](notebooks/lrf.ipynb). Здесь уже сложнее с определением диапазона lr, поэтому только на основе предыдущего опыта и эксперименты.

##### 4) Файнтюнинг всей сети.
```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/movie-genre-classification python train.py configs/train_full.py
```

##### 5) SWA (опционально).
Запускаем скрипт:
```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/movie-genre-classification python swa.py configs/train_full.py
```
При текущей конфигурации, мне не удалось увеличить качество за счёт SWA, но попробовать всегда стоит!

##### 5) Запуски с другими конфигами (опционально).

Тренировка с FocalLoss:
```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/movie-genre-classification python swa.py configs/train_full_with_focal.py
```

Тренировка с LabelSmoothing:
```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/movie-genre-classification python swa.py configs/train_full_with_smoothing.py
```

### Конвертация в TorchScript

Пример конвертации модели в TorchScript находится в [тетрадке](notebooks/to-script.ipynb).
При использовании данного способа выноса модели в прод, рекомендуется вынести код в отдельный скрипт или вообще в пайплайн тренировки.