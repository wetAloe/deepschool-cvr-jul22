import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class ImageNetEvaluate:

    def __init__(
            self,
            imagenet_validation_dataloader: DataLoader,
            model: nn.Module,
            device: str,
    ):
        self.imagenet_validation_dataloader = imagenet_validation_dataloader
        self.model = model
        self.device = torch.device(device)

    def evaluate(self):
        self.model.eval()
        self.model.to(self.device)
        total_correct = 0
        total_images = 0
        tqdm_iter = tqdm(self.imagenet_validation_dataloader)
        with torch.no_grad():
            for images, labels in tqdm_iter:
                output_tensor = self.model(images.to(self.device))
                probability = torch.softmax(output_tensor, dim=-1).detach().cpu().numpy()
                prediction = np.argsort(probability)[:, ::-1]
                top_1 = prediction[:, 0]
                labels = labels.numpy()
                correct = np.sum(top_1 == labels)
                total_correct += correct
                total_images += images.size(0)
                tqdm_iter.set_description(f'batch accuracy {(total_correct / total_images):.5f}')

        top_1_accuracy = total_correct / total_images
        return top_1_accuracy
