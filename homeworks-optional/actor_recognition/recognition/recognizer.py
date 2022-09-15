import insightface
import numpy as np
from pathlib import Path
import typing as t
from .types import RGBImage, Descriptor, BoundingBox, Landmarks
from .utils import get_session


# TODO: you can pass a trained refinement network here
class FaceRecognizer:
    def __init__(
            self,
            detector_model: Path,
            embedding_model: Path,
            input_size: int = 320,
            refinement_function: t.Optional[t.Callable[[Descriptor], Descriptor]] = None
    ):
        self.detector = insightface.model_zoo.scrfd.SCRFD(session=get_session(detector_model))
        self.detector.prepare(ctx_id=0, nms_thresh=0.4, det_thresh=0.2, input_size=(input_size, input_size))
        self.embedder = insightface.model_zoo.arcface_onnx.ArcFaceONNX(
            embedding_model,
            session=get_session(embedding_model)
        )
        if refinement_function is None:
            self.refinement_function = lambda x: x
        else:
            self.refinement_function = refinement_function

    def detect(self, img: RGBImage) -> t.Tuple[t.Sequence[BoundingBox], t.Sequence[Landmarks]]:
        return self.detector.detect(img, max_num=30)

    def get_embedding(
            self,
            face_img: RGBImage,
            bounding_box: t.Optional[BoundingBox] = None,
            landmarks: t.Optional[Landmarks] = None
    ):
        if bounding_box is None or landmarks is None:
            res = self.detect(face_img)
            bounding_box = res[0][0]
            landmarks = res[1][0]
        face = insightface.app.common.Face(bbox=bounding_box, kps=landmarks)
        return self.refinement_function(self.embedder.get(face_img, face))
