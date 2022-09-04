import numpy as np
import torch
from torch import nn
from torch.quantization.quantize_fx import fuse_fx
from torch.utils.data import DataLoader

from model.mobilenet_v2 import MobileNetV2
from tiny_imagenet.tiny_imagenet_dataset import CsvDataset
from tiny_imagenet.tiny_imagenet_dataset import get_valid_transform
from tiny_imagenet.tiny_imagenet_evaluate import ImageNetEvaluate

MODEL_PATH = './resources/mobilenet_v2_s1_ti_regular.pth'
VALIDATION_CSV_PATH = './resources/tiny-imagenet-200/validation.csv'
IMAGENET_ROOT_DIR = './resources/tiny-imagenet-200'
INPUT_IMAGE_SIZE = 64

# Тут 2 отличия от обычного мобайлнета.
# 1. У первого конва страйд 1, потому что картинка на входе и так маленькая
# 2. Используем ReLU вместо ReLU6.
model = MobileNetV2(
    num_classes=200,
    first_conv_stride=1,
    activation_layer=nn.ReLU,
)

mean = np.array([0.406, 0.456, 0.485], dtype=np.float32)
std = np.array([0.225, 0.224, 0.229], dtype=np.float32)

dataset = CsvDataset(
    csv_file=VALIDATION_CSV_PATH,
    data_dir=IMAGENET_ROOT_DIR,
    transform=get_valid_transform(input_size=INPUT_IMAGE_SIZE, mean=mean, std=std)
)

dataloader = DataLoader(
    dataset, batch_size=64, num_workers=8, shuffle=False,
)

if __name__ == '__main__':

    model.load_state_dict(
        torch.load(MODEL_PATH)
    )

    model.eval()
    # Ещё один халявный способ сфьюзить батчнормы
    model = fuse_fx(model)
    validator = ImageNetEvaluate(
        imagenet_validation_dataloader=dataloader,
        model=model,
        device='cpu'  # Модельки будем считать на cpu, потому что для квантования на видюхах нужен tensorrt например.
    )
    # Можно и по выходу tqdm смотреть что модель ускорилась
    # Но правильно нужно мерить модельку полностью отключив от пайплайна всё кроме прогона картинки через сетку
    # Т.е. Просто создать тензор, и в цикле только его крутить через модельку
    validator.evaluate()
    # batch accuracy 0.70650: 100%|█████████████████| 157/157 [00:29<00:00,  5.30it/s] Это если фьюзить батчнормы
    # batch accuracy 0.70650: 100%|█████████████████| 157/157 [00:39<00:00,  4.01it/s] А это если не фьюзить
    # В 1.3 раза быстрее на халяву!
