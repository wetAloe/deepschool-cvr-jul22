import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model.mobilenet_v2 import MobileNetV2
from quantization_utils.static_quantization import quantize_static
from tiny_imagenet.tiny_imagenet_dataset import CsvDataset
from tiny_imagenet.tiny_imagenet_dataset import get_valid_transform
from tiny_imagenet.tiny_imagenet_evaluate import ImageNetEvaluate

MODEL_PATH = './resources/mobilenet_v2_s1_ti_regular.pth'
VALIDATION_CSV_PATH = './resources/tiny-imagenet-200/validation.csv'
IMAGENET_ROOT_DIR = './resources/tiny-imagenet-200'
INPUT_IMAGE_SIZE = 64


model = MobileNetV2(
    num_classes=200,
    first_conv_stride=1,
    activation_layer=nn.ReLU
)

mean = np.array([0.406, 0.456, 0.485], dtype=np.float32)
std = np.array([0.225, 0.224, 0.229], dtype=np.float32)

dataset = CsvDataset(
    csv_file=VALIDATION_CSV_PATH,
    data_dir=IMAGENET_ROOT_DIR,
    transform=get_valid_transform(input_size=INPUT_IMAGE_SIZE, mean=mean, std=std)
)

dataloader = DataLoader(
    dataset, batch_size=64, num_workers=8, shuffle=True,
)

if __name__ == '__main__':

    model.load_state_dict(
        torch.load(MODEL_PATH)
    )

    model = quantize_static(model, dataloader)
    validator = ImageNetEvaluate(
        imagenet_validation_dataloader=dataloader,
        model=model,
        device='cpu'
    )

    validator.evaluate()
    # Вот это реально злодейский буст по скорости. в 6 раз
    # А если ещё учитывать фьюзинг то это 8 раз. ОГОНЬ!
    # batch accuracy 0.62980: 100%|█████████████████| 157/157 [00:05<00:00, 29.81it/s]
