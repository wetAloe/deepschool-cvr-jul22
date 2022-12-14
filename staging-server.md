# Стейджинг-сервер

На курсе для деплоя и демок ваших приложений мы будем использовать этот сервачок: **91.206.15.25**

(но если у кого-то есть свой сервак с публичным ip, то можно и на своём, как удобно.) 

На этом серваке есть только CPU.

### Немного правил:

Наш стейджинг-сервер это небольшая общага. Чтобы случайно не навредить другому студенту, предлагаю
придерживаться пары простых правил:

#### Порты
Чтобы избежать коллизии портов между пользователями, в [табличке]((https://docs.google.com/spreadsheets/d/1mQSsBWeq29IGiwqAXKfsON8lx-2yQVTdM5r7EnPV8eg/edit?usp=sharing))
указано, какому пользователю какие порты можно использовать.
Если у моего пользователя стоит `**01` это значит, что для портов своих приложений я могу использовать все четырехзначные
порты, которые заканчиваются на `01` (1001, 1101, ... , и так примерно сто раз)

#### Названия контейнеров

Названия контейнеров нужно начинать с префикса в виде никнейма из [таблички]((https://docs.google.com/spreadsheets/d/1mQSsBWeq29IGiwqAXKfsON8lx-2yQVTdM5r7EnPV8eg/edit?usp=sharing))

т.е. если мой ник `anton_srk`, то свой контейнер я, например, назову `anton_srk_genres_service`

### Как подключиться:

```bash
ssh username@91.206.15.25
# и вводим пароль
```

`username` и пароль можно взять в [этой](https://docs.google.com/spreadsheets/d/1mQSsBWeq29IGiwqAXKfsON8lx-2yQVTdM5r7EnPV8eg/edit?usp=sharing) табличке на листе staging-сервер

Копипастить пароль каждый раз неудобно, поэтому рекомендуем завести ssh-ключ для этого сервака.

[Как сгенерировать ssh-ключ](https://selectel.ru/blog/tutorials/how-to-generate-ssh/)

Далее нужно добавить публичную часть ключа на сервер:
```bash
ssh-copy-id -i /path/to/your/id_rsa.pub username@91.206.15.25
```

Теперь можно проверить, что стало пускать без пароля.

### Про ssh-ключи и gitlab:

если ssh-ключ закинуть в переменную(или файл) гитлаба как есть, то гитлаб его испортит (понавтыкает /n в разных местах)
и по такому испорченному ключу сервер вас не запустит.

Есть простой хак:

Можно закодировать ssh-ключ в base64-строку:

```bash
base64 /path/to/your/ssh_private_key
```

И эту строчку уже занести в переменные гитлаба

а когда в CI-пайплайнах нам понадобится ключ, просто декодируем его и записываем в файлик:

```bash
echo "$HOST_SSH_KEY_BASE64" | base64 -d > ~/.ssh/id_rsa
```

### Про dvc

Использовать dvc с гуглдрайвом для приложений, которые мы куда-то автоматически деплоим
плохая идея. Если помните, то там нужно перейти по ссылке, ввести ключ и т.п. Такое не подходит
для автоматизации.

Будем использовать для dvc наш стейджинг. dvc будет просто подключаться к нашему серваку
по ssh и в папке `/home/<your_username>/<your_name_for_dvc_dir>` писать и спрашивать файлики.

#### Как этого достичь:

1) поставить dvc, который умеет ходить по ssh:

```bash
pip install dvc[ssh]
```

2) инициализировать dvc в вашем репозитории:

```bash
### DVC_REMOTE_NAME -- просто то, как вы хотите назвать хост для dvc
### USERNAME -- username из таблички
dvc remote add --default $(DVC_REMOTE_NAME) ssh://91.206.15.25/home/$(USERNAME)/dvc_files
dvc remote modify $(DVC_REMOTE_NAME) user $(USERNAME)
dvc config cache.type hardlink,symlink
```

После такой настройки dvc будет складывать все файлики на стейджинг-серваке в вашей рабочей
директории в папке `dvc_files`

Теперь можете потестить dvc (добавить файлики, пушнуть и всё вот это:))

Если будет ругаться на то что нет прав на доступ, можно попытаться еще дополнительно
указать ему путь до приватного ключа

```bash
dvc remote modify $(DVC_REMOTE_NAME) keyfile /path/to/your/private_key
```

### Про ansible

Если ваш ssh-ключ лежит в каком-то кастомном месте (не в `~/.ssh`), то
ансиблу нужно указать путь до ключа:


```bash
ansible-playbook -i ${INVENTORY_PATH} ${PLAYBOOK}  \
    -e your=first_var \
    -e your=second_var \
    --key-file ${PATH_TO_SSH_KEY}
```

