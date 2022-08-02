import types
import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.ops as ops
from glob import glob


class ExternalInputIterator:
    def __init__(self, batch_size: int):
        self.fnames = glob('/workspace/project/data/images/*.jpg')
        self.fnames.remove('/workspace/project/data/images/broken_image.jpg')
        self.batch_size = batch_size

    def __iter__(self):
        self.i = 0
        self.n = len(self.fnames)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            filename_jpg = self.fnames[self.i]
            label = np.random.randint(0, 2, size=(5,)).astype(np.float32)
            with open(filename_jpg, 'rb') as f:
                batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            labels.append(label)
            self.i = (self.i + 1) % self.n
        return batch, labels


class CustomPipeline(Pipeline):
    def __init__(self, batch_size: int, num_threads: int, device_id: int):
        super().__init__(batch_size=batch_size, num_threads=num_threads, device_id=device_id)

        self.input = ops.ExternalSource(source=ExternalInputIterator(batch_size), num_outputs=2)
        self.decoder = ops.ImageDecoder(device='mixed', output_type=types.RGB)
        self.resize = ops.Resize(device='gpu', size=(224, 224), interp_type=types.INTERP_TRIANGULAR)
        self.normalization = ops.CropMirrorNormalize(
            device="gpu",
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decoder(jpegs)
        images = self.resize(images)
        images = self.normalization(images)

        return images, labels.gpu()


if __name__ == '__main__':
    pipeline = CustomPipeline(batch_size=60, num_threads=8, device_id=0)
    pipeline.build()
    output = pipeline.run()
