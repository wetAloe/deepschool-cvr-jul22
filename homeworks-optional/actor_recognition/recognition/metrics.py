import numpy as np

from .types import Descriptor


def cosine_similarity(left_descriptor: Descriptor, right_descriptor: Descriptor) -> float:
    left_normalized = left_descriptor / np.linalg.norm(left_descriptor)
    right_normalized = right_descriptor / np.linalg.norm(right_descriptor)
    return left_normalized @ right_normalized


def cosine_distance(left_descriptor: Descriptor, right_descriptor: Descriptor) -> float:
    return 1 - cosine_similarity(left_descriptor, right_descriptor)

