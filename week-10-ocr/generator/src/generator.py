import numpy as np
import PIL
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont, Image
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, Union
from glob import glob
import os
import cv2
import albumentations as alb


def render_text(
    text: str, 
    font: Union[ImageFont.FreeTypeFont, Callable[[], ImageFont.FreeTypeFont]],
    color: Union[Tuple[int, int, int], Callable[[], Tuple[int, int, int]]] = (0, 0, 0),
    spacing: Union[float, Callable[[], float]] = 0.7,
    sp_var: Union[float, Callable[[], float]] = 0.05,
    alpha: Union[int, Callable[[], int]] = 255,
    pad: int = 1,
    transforms: Optional[alb.BasicTransform] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Рендерит text в картинку.

    Возвращает два объекта: отрендеренную картинку и маску.
    Маска обозначает какие пиксели на картинке фон, а какие являются отрендеренным
    значением (например, если value - строка текста, то маска должна отражать позицию букв
    на отрендеренной картинке). Маска имеет такой же spatial size, как и картинка.

    :param text: значение, которое требуется отрендерить в картинку
    :param font: шрифт для рендера текста
    :param color: цвет генерируемого текста (default (0, 0, 0))
    :param spacing: среднее значение расстояния между буквами (default 0.7)
    :param sp_var: параметр, отвечающий за вариативность расстояния между буквами расстояние - случайная (default 0.05)
    величина из равномерного распределения Uni[spacing - sp_var * font_size, spacing + sp_var * font_size],
    где font_size - размер шрифта, диктуемый writer-ом
    :param alpha: значение, которым будет заполнена маска в месте букв (default 255)
    :param pad: отступ от каждого края при рендере текста (в пикселях) (default 255)
    :param transforms: трансформации, которые будут применены маске (default None)
    :return: картинку, маску
    """
    spacings = np.random.uniform(
        spacing - sp_var * font.size,
        spacing + sp_var * font.size,
        size=len(text),
    )
    text_w = int(sum(font.getsize(c)[0] for c in text) + sum(spacings[:-1]))
    text_h = int(font.getsize(text)[1])

    image = Image.new(
        'RGBA',
        (text_w + 2 * pad, text_h + 2 * pad),
        tuple(list((0, 0, 0)) + [0]),
    )
    draw = ImageDraw.Draw(image)

    x = pad
    for c, sp in zip(text, spacings):
        w, _ = font.getsize(c)
        draw.text((x, pad), c, tuple(color), font=font)
        x = x + w + sp

    # apply transforms to 4-channels image
    image = np.asarray(image).astype(np.float32)

    # cast to PIL.Image
    mask = image[..., 3]
    mask = mask / max(mask.max(), 1) * alpha
    mask = mask.astype(np.uint8)
    mask = transforms(image=mask)['image'] if transforms else mask

    image = image[..., :3].astype(np.uint8)
    return image, mask


def load_fonts(fonts_dir: str, sizes: List[int]) -> Optional[List[ImageFont.FreeTypeFont]]:
    """Загружает шрифты, возвращает список с каждым шрифтом каждого размера.

    Результат можно представить как декартово произведение каждого штрифта с каждым размером.

    :param zip_name: имя zip файла
    :param sizes: список с требуемыми размерами шрифтов
    """
    to_fonts = glob(os.path.join(fonts_dir, '*'))
    sizes = np.repeat(sizes, len(to_fonts))
    to_fonts = np.tile(to_fonts, len(sizes))
    return [ImageFont.truetype(font, size) for font, size in zip(to_fonts, sizes)]


def overlay(
    back_img,
    front_img,
    front_mask,
    pad: Union[Tuple[int, int, int, int], Callable[[], Tuple[int, int, int, int]]] = (0, 0, 0, 0),
    transforms: Optional[alb.BasicTransform] = None,
    alpha: Union[int, Callable[[], int]] = 255,
    fit_mode: Optional[str] = 'background',
) -> Tuple[np.ndarray, np.ndarray]:
    """Рендерит value в картинку с помощью генератора переднего плана.

    Возвращает два объекта: отрендеренную картинку и маску.
    Маска обозначает какие пиксели на картинке фон, а какие являются отрендеренным
    значением (например, если value - строка текста, то маска должна отражать позицию букв
    на отрендеренной картинке). Маска имеет такой же spatial size, как и картинка.
    
    :param back_img: 
    :param front_img: 
    :param front_mask: 
    :param pad: отступы от каждой стороны фона при наложении текста (default (0, 0, 0, 0))
    :param transforms: трансформации, которые будут применены к 4-х канальной картинке (rgb + alpha) (default None)
    :param alpha: значение, которым будет заполнена маска (default 255)
    :param fit_mode: указывает что ресайзим перед наложением "background" или "foreground", (default None)
    :return: картинку, маску
    """
    lpad, tpad, rpad, bpad = pad

    bh, bw = back_img.shape[:2]
    fh, fw = front_img.shape[:2]

    if fit_mode is None:
        # выбираем область фона, куда будем вставлять картинку
        assert fw + lpad + rpad <= bw, 'Картинка переднего плана не вмещается с указанными паддингами'
        assert fh + tpad + bpad <= bh, 'Картинка переднего плана не вмещается с указанными паддингами'

        x_left = np.random.randint(0, bw - fw - lpad - rpad)
        y_left = np.random.randint(0, bh - fh - tpad - bpad)
        x_right = x_left + lpad + fw + rpad
        y_right = y_left + tpad + fh + bpad
        back_img = back_img[y_left:y_right, x_left:x_right, ...]
    if fit_mode == 'background':
        back_img = cv2.resize(back_img, (fw + rpad + lpad, fh + tpad + bpad))
    if fit_mode == 'foreground':
        front_img = cv2.resize(front_img, (bw, bh))
        front_mask = cv2.resize(front_mask, (bw, bh))

    # подрезаем foreground если где-то падинги отрицательные
    front_img = front_img[
        max(0, -tpad) : max(fh, fh - bpad), max(0, -lpad) : max(fw, fw - rpad)
    ]
    front_mask = front_mask[
        max(0, -tpad) : max(fh, fh - bpad), max(0, -lpad) : max(fw, fw - rpad)
    ]

    # накладываем текст поверх фона
    front_mask_c = np.expand_dims(front_mask, -1) / 255.0
    back_img[
        max(0, tpad) : tpad + fh, max(0, lpad) : lpad + fw, :
    ] = front_img * front_mask_c + back_img[
        max(0, tpad) : tpad + fh, max(0, lpad) : lpad + fw, :
    ] * (
        1 - front_mask_c
    )

    # накладываем маску текста поверх маски фоновой картинки
    mask = np.zeros_like(back_img[:, :, 0])
    mask[
        max(0, tpad) : tpad + fh, max(0, lpad) : lpad + fw
    ] = front_mask

    # объединяем картинку и маску в одну 4=канальную картинку
    img = cv2.merge((back_img, mask)).astype(np.float32)
    img = transforms(image=img)['image'] if transforms else img

    # разделяем картинку и маску
    img = img.astype(np.uint8)
    image = img[..., :3]
    mask = img[..., 3]

    return image, mask
