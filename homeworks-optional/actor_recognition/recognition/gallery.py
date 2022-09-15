from pathlib import Path
from PIL import Image as PILImage
import numpy as np
import typing as t
from .recognizer import FaceRecognizer
from .metrics import cosine_similarity
from .types import Descriptor, Person, GalleryFinder


def config_gallery_finder(
        face_recognizer: FaceRecognizer,
        gallery_root: Path,
        similarity_threshold: float = 0.3
) -> GalleryFinder:
    gallery: t.List[Person] = []
    for cur_person_dir in gallery_root.iterdir():
        cur_name = cur_person_dir.stem
        cur_descriptors = [face_recognizer.get_embedding(np.asarray(PILImage.open(str(cur_img))))
                           for cur_img in
                           filter(lambda p: p.suffix in ['.png', '.jpg', '.jpeg'], cur_person_dir.iterdir())]
        gallery.append(Person(name=cur_name, descriptors=cur_descriptors))
    compressed_gallery = [Person(name=cur_person.name, descriptors=[np.mean(cur_person.descriptors, axis=0)])
                          for cur_person in gallery]

    def find(query_descriptor: Descriptor) -> t.Optional[Person]:
        similarities = [cosine_similarity(query_descriptor, cur_person.descriptors[0])
                        for cur_person in compressed_gallery]
        max_idx = np.argmax(similarities)
        max_sim = similarities[max_idx]
        if max_sim > similarity_threshold:
            return gallery[max_idx]
        else:
            return None

    return find
