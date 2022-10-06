import torch
from torchvision import transforms
from ts.torch_handler.vision_handler import VisionHandler

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ModelHandler(VisionHandler):
    image_processing = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    def initialize(self, context):
        super().initialize(context)
        self._classes = self.model.classes

    def postprocess(self, probs: torch.Tensor):
        res = []
        for pr in probs:
            res.append(
                {self._classes[i]: float(pr[i]) for i in range(len(pr))}
            )
        return res
