from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import typing as t


KwArgs = t.Dict[str, t.Any]
RGBImage = t.Union[NDArray[np.uint8], NDArray[np.float32]]  # a HxWxC tensor representing an image with RGB channels
Descriptor = NDArray[np.float32]  # a float vector containing a face descriptor
BoundingBox = t.Union[t.Tuple[int, int, int, int], NDArray[np.int32]]  # a bounding box in XYXY format
Landmarks = NDArray[np.int32]


@dataclass
class Person:
    name: str
    descriptors: t.Sequence[Descriptor]


@dataclass
class Track:
    track_id: int
    descriptors: t.Sequence[Descriptor]
    person: t.Optional[Person]
    tracking_box: BoundingBox


GalleryFinder = t.Callable[[RGBImage], Person]
Metric = t.Callable[[Descriptor, Descriptor], float]
