from copy import deepcopy
from pathlib import Path
from typing import Dict
from typing import Tuple
from typing import Union

import torch
from torch import nn
from torch.quantization.quantize_fx import convert_fx
from torch.quantization.quantize_fx import fuse_fx
from torch.quantization.quantize_fx import prepare_qat_fx
from torch.utils.data import DataLoader

from train_utils.distillate import ImageNetDistiller

# Этот молодой человек отвественнен за выбор схемы квантования
# Он говорит нам какие узлы Fake квантования нам навесить
# Они также как и обсёрверы просто навешиваются на веса и активации
DEFAULT_Q_CONFIG_DICT = {"": torch.quantization.get_default_qat_qconfig('fbgemm')}
# В точре по умолчанию два бекенда
# Один для мобилок, а другой для серверов.
# Отличие одно, торчовые квантованные сетки на мобилках не умеют в векторное квантование весов


def quantize_with_qat(
        model: nn.Module,
        data_loader: DataLoader,
        log_dir: Union[Path, str],
        q_config_dict: Dict = None,
        device: str = 'cuda:0'
) -> Tuple[nn.Module, nn.Module]:
    """ Quantize and train network with distillation using quantization aware training

    Parameters
    ----------
    model:
        Model to be quantized
    data_loader:
        Data Loader with Training data.
    log_dir:
        Log dir for checkpoints and tensorboard logs
    q_config_dict:
        quantization config dict
    device:
        device for training

    Returns
    -------
    :
        Quantized and trained model.
    """
    prepared_model = deepcopy(model)
    q_config_dict = q_config_dict if q_config_dict is not None else DEFAULT_Q_CONFIG_DICT
    # Вся магия происходит здесь
    # Torch FX позволяет нам вместо monkey patching
    # Редактировать граф во время рантайма
    # А значит мы можем вставить перед нодой нашу ноду для фейк квантования

    # По умолчанию батчнормы не фьюязтся
    # Но их наличие славно подпортит нам жизнь, потому что из-за них
    # У нас получается сильно несоотвествие между Fake Quant моделью и Quant моделью
    # Есть много способов решить эту проблему, например сделать сильное усреднение для moving average
    # Но я предпочитаю убирать, если это получается
    prepared_model.eval()
    prepared_model = fuse_fx(prepared_model)
    prepared_model.train()
    model.eval()

    prepared_model = prepare_qat_fx(
        model=prepared_model,
        qconfig_dict=q_config_dict,
    )
    # До FX все подобные трюки использовали monkey patching
    # И Тайные знания о структуре модели, некоторые писали свои трейсеры.
    # Нормально квантовать, не зная как связаны между собой слои, не получится.
    # Например пайторчи предлагали явно самим встраивать узлы квантования и деквантования в начале и в конце сети
    # Потому что автоматически понять что это нужно сделать нельзя было.
    imagenet_distiller = ImageNetDistiller(
        quantized_model=prepared_model,
        regular_model=model,
        train_data_loader=data_loader,
        logdir=log_dir,
        scheduler_params={
            'T_0': int(len(data_loader) / 2),
        },
        device=device,
        number_of_epoch=2,
    )

    imagenet_distiller.train()

    quantized_model = deepcopy(prepared_model)
    # Собственной в данной строчке и происходит ускорение и квантование
    # Моделька до этого была обычной, просто собирала необходимые статистики
    # А вот теперь мы заменяем операции на квантованные в int8
    quantized_model = convert_fx(quantized_model)
    # Мы вернём две модельки. Чтобы показать что это нормально что Fake Quant и Quant модельки отличаются.
    return quantized_model, prepared_model
