## Толока

Презентация по Толоке живет [здесь](https://docs.google.com/presentation/d/1efDYWJASzbCXKwwldwpZrZYUtFkB81iV9q4ZfiiN5sY/edit?usp=sharing).


### Что  нужно сделать

1. Сделать в Песочнице следующие проекты:
   1. Сбор фотографий со штрих-кодами
   2. Получение боксов со штрих-кодами
   3. Распознавание цифр под штрих-кодами
2. Добавить e-mail **lizmisha@yandex.ru** в доверенные пользователи
(_Пользователи_ -> _Добавить доверенных пользователей_ -> _Добавить пользователя_)
3. После успешного ревью проектов в Песочнице перенести их из Песочницы в Толоку
4. Перепроверить настройки проектов уже на самой Толоке
5. Запустить проекты и на выходе получить готовый датасет


### Подробнее о каждом проекте в Толоке

**ВАЖНО:** отложенную проверку каждого проекта осуществляете вы сами. То есть,
например, вы сами должны просмотреть фотографии со штрих-кодами, которые
вам прислали, и решить принимать их или нет.

#### Сбор фотографий со штрих-кодами

Для создания можно воспользоваться стандартным шаблоном "Пешеходные
задания". Только нужно будет убрать проверку геопозиции и комментарий из
интерфейса задания.

**Цель проекта:** нужно получить 500 фотографий со штрих-кодами. Под штрих-кодами
обязательно должны быть цифры (их мы будем распознавать в крайнем
проекте). Фотография штрих-кода должна быть сделана с близкого или среднего
расстояния. Штрих-код и цифры должны быть видны целиком.

**Обратите внимание:** в конечном итоге у вас должно быть 500+-15 изображений.

**Основные параметры проекта**:
- **Цена**: 0.005
- **Заданий на странице** (настраивается при загрузке заданий в пул): 1
- **Перекрытие**: 1
- Должна быть **отложенная приемка**

#### Получение боксов со штрих-кодами

Для создания можно воспользоваться стандартным шаблоном "Выделение объектов
на изображении".

**Цель проекта:** нужно выделить штрих-код вместе с цифрами под ними. Важно
обратить внимание, что это должен быть один бокс. Не нужно отдельно выделять
штрих-код, а отдельно цифры под ними.

**Основные параметры проекта**:
- **Цена**: 0.005
- **Заданий на странице** (настраивается при загрузке заданий в пул): 1
- **Перекрытие**: 1
- Должна быть **отложенная приемка**

#### Распознавание цифр под штрих-кодами

Для создания можно воспользоваться стандартным шаблоном "Распознавание
текста с изображения (OCR)".

**Цель проекта:** распознать цифры под штрих-кодом. Важно, чтобы разные
группы цифр отделялись только одним пробелом, а не несколькими.

**Основные параметры проекта**:
- **Цена**: 0.01
- **Заданий на странице** (настраивается при загрузке заданий в пул): 15
- **Перекрытие**: 1
- Должна быть **отложенная приемка**
- Должны быть **бесплатные обучающий и экзаменационный пулы**. На каждый пул
можно разметить самому по 5 примеров.