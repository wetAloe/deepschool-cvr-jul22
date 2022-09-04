import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model.mobilenet_v2 import MobileNetV2
from quantization_utils.fake_quantization import quantize_with_qat
from tiny_imagenet.tiny_imagenet_dataset import CsvDataset
from tiny_imagenet.tiny_imagenet_dataset import get_valid_transform
from tiny_imagenet.tiny_imagenet_evaluate import ImageNetEvaluate

MODEL_PATH = './resources/mobilenet_v2_s1_ti_regular.pth'
VALIDATION_CSV_PATH = './resources/tiny-imagenet-200/validation.csv'
TRAIN_CSV_PATH = './resources/tiny-imagenet-200/train.csv'
IMAGENET_ROOT_DIR = './resources/tiny-imagenet-200'
LOG_DIR = './log_dir'
INPUT_IMAGE_SIZE = 64


model = MobileNetV2(
    num_classes=200,
    first_conv_stride=1,
    activation_layer=nn.ReLU,
)

mean = np.array([0.406, 0.456, 0.485], dtype=np.float32)
std = np.array([0.225, 0.224, 0.229], dtype=np.float32)

valid_dataset = CsvDataset(
    csv_file=VALIDATION_CSV_PATH,
    data_dir=IMAGENET_ROOT_DIR,
    transform=get_valid_transform(input_size=INPUT_IMAGE_SIZE, mean=mean, std=std)
)

train_dataset = CsvDataset(
    csv_file=TRAIN_CSV_PATH,
    data_dir=IMAGENET_ROOT_DIR,
    transform=get_valid_transform(input_size=INPUT_IMAGE_SIZE, mean=mean, std=std),
)

valid_dataloader = DataLoader(
    valid_dataset, batch_size=64, num_workers=8, shuffle=True,
)

train_dataloader = DataLoader(
    train_dataset, batch_size=64, num_workers=8, shuffle=True,
)

if __name__ == '__main__':

    model.load_state_dict(
        torch.load(MODEL_PATH)
    )

    model_quant, model_fake_quant = quantize_with_qat(
        model,
        train_dataloader,
        log_dir=LOG_DIR,
    )
    validator = ImageNetEvaluate(
        imagenet_validation_dataloader=valid_dataloader,
        model=model_quant,
        device='cpu'
    )

    validator.evaluate()

    validator = ImageNetEvaluate(
        imagenet_validation_dataloader=valid_dataloader,
        model=model_fake_quant,
        device='cpu'
    )

    validator.evaluate()
    # batch accuracy 0.69140: 100%|██████████| 157/157 [00:05<00:00, 29.62it/s] Quantized
    # batch accuracy 0.69650: 100%|██████████| 157/157 [00:56<00:00,  2.77it/s] Fake Quantized
    # With ReLU6 (Потому что ReLU6 не фьюзится)
    # batch accuracy 0.68980: 100%|██████████| 157/157 [00:08<00:00, 17.55it/s] Quantized
    # batch accuracy 0.69140: 100%|██████████| 157/157 [01:08<00:00,  2.30it/s]] Fake Quantized
