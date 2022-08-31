from typing import Tuple
import numpy as np
import cv2
import torch


def torch_preprocessing(
    image: np.ndarray, image_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Convert numpy-image array for inference Torch model.
    """
    # resize
    image = cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_LINEAR)

    # normalize
    mean = np.array((0.485, 0.456, 0.406), dtype=np.float32) * 255.0
    std = np.array((0.229, 0.224, 0.225), dtype=np.float32) * 255.0
    denominator = np.reciprocal(std, dtype=np.float32)
    image = image.astype(np.float32)
    image -= mean
    image *= denominator

    # to tensor and transpose
    image = torch.from_numpy(image.transpose((2, 0, 1)))[None]
    return image


def trt_preprocessing(
    image: np.ndarray, image_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Convert numpy-image to array for inference TensorRT model.
    """

    # resize
    image = cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_LINEAR)

    # normalize
    mean = np.array((0.485, 0.456, 0.406), dtype=np.float32) * 255.0
    std = np.array((0.229, 0.224, 0.225), dtype=np.float32) * 255.0
    denominator = np.reciprocal(std, dtype=np.float32)
    image = image.astype(np.float32)
    image -= mean
    image *= denominator

    # transpose
    image = np.array(image.transpose((2, 0, 1))[None], dtype=np.float32, order="C")
    return image
