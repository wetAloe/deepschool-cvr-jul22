import numpy as np
from PIL import Image as PILImage
from time import time
import logging
import typing as t

from .drawing import draw_tracks
from .tracking import Sort
from .recognizer import FaceRecognizer
from .types import RGBImage, BoundingBox, Track, GalleryFinder


class FaceTracker:

    def __init__(
            self,
            face_recognizer: FaceRecognizer,
            gallery_find_function: t.Optional[GalleryFinder] = None,
            input_size=320,
            detect_frame=1,
            min_x=0,
            max_x=np.inf,
            min_y=0,
            max_y=np.inf
    ):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.detect_frame = detect_frame
        self.gallery_find_function = gallery_find_function
        self.num_frames = 0
        self.min_size = 36
        self.input_size = input_size
        self.face_recognizer = face_recognizer
        self.tracker = Sort(max_age=5, min_hits=10)
        self.times = []

    def process_face(self, frame: RGBImage, bbox: BoundingBox, landmarks: np.ndarray):
        return self.face_recognizer.get_embedding(frame, bbox, landmarks)

    def find_actors_for_tracks(self, tracks: t.Sequence[Track]) -> None:
        if self.gallery_find_function is None:
            return None
        for cur_track in tracks:
            # TODO: you may want to change the next line
            cur_track.person = self.gallery_find_function(cur_track.descriptors[-1])

    def process_frame(self, frame: RGBImage, width: int, height: int) -> RGBImage:
        start = time()
        img = PILImage.fromarray(frame)
        bboxes, landmarks = self.face_recognizer.detect(frame)
        bbox_size = 0.5 * (bboxes[:, 2] - bboxes[:, 0] + bboxes[:, 3] - bboxes[:, 1])
        size_logic = bbox_size > self.min_size
        x_logic = np.logical_and(bboxes[:, 0] > self.min_x, bboxes[:, 2] < self.max_x)
        y_logic = np.logical_and(bboxes[:, 1] > self.min_y, bboxes[:, 3] < self.max_y)
        filter_logic = np.logical_and(np.logical_and(x_logic, y_logic), size_logic)
        filtered_bboxes = bboxes[filter_logic]
        filtered_landmarks = landmarks[filter_logic]
        additional_info = [self.process_face(frame, cur_bbox, cur_landmarks)
                           for cur_bbox, cur_landmarks in zip(filtered_bboxes, filtered_landmarks)]
        tracks = self.tracker.update(filtered_bboxes, np.asarray((height, width)), additional_info, self.detect_frame)
        self.find_actors_for_tracks(tracks)
        self.times.append(time() - start)
        img = draw_tracks(img, tracks)
        self.num_frames += 1
        return np.asarray(img)
