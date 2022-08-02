import cv2
import numpy as np
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2


class TorchDataset(Dataset):
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.fnames = glob('/workspace/project/data/images/*.jpg')
        self.fnames.remove('/workspace/project/data/images/broken_image.jpg')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        image = Image.open(self.fnames[idx]).convert('RGB')
        label = np.random.randint(0, 2, size=(5,)).astype(np.float32)

        image = self.transforms(image)
        image = np.array(image)

        return {'image': image, 'label': label}


class CV2Dataset(Dataset):
    def __init__(self):
        self.transforms = albu.Compose([
            albu.Resize(224, 224, ),
            albu.Normalize(),
            ToTensorV2(),
        ])
        self.fnames = glob('/workspace/project/data/images/*.jpg')
        self.fnames.remove('/workspace/project/data/images/broken_image.jpg')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        image = cv2.imread(self.fnames[idx])[..., ::-1]
        label = np.random.randint(0, 2, size=(5,)).astype(np.float32)

        image = self.transforms(image=image)['image']
        image = np.array(image)

        return {'image': image, 'label': label}
