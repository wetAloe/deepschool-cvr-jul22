## Для запуска всех примеров рекомендую использовать [NGC контейнеры](https://ngc.nvidia.com/catalog)

### 1. Билдим и запускаем контейнер
````
docker build -t dali .
chmod +x run.sh  
./run.sh  
````
### 2. Запускаем jupyter и копируем token
````
jupyter-notebook
````
пример токена `6681f809d614c362b4837a37e5649ac6c8d85f253c9ecc4d`

### 3. В браузере запускаем `http://localhost:8888/` и вставляем токен.

### 4. Присоединиться к запущенному контейнеру
````
docker exec -it dali /bin/bash
````