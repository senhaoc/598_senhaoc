import os
from typing import List, Tuple
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from config import DATA_ROOT


def get_image_paths(
    root: str = DATA_ROOT,
    split: str = "train",
    max_samples: int | None = None,
) -> Tuple[List[str], np.ndarray]:
    """Return image paths and labels (0: NORMAL, 1: PNEUMONIA)."""
    paths: list[str] = []
    labels: list[int] = []
    for label_idx, cat in enumerate(["NORMAL", "PNEUMONIA"]):
        folder = os.path.join(root, split, cat)
        if not os.path.exists(folder):
            continue
        imgs = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if max_samples is not None:
            np.random.shuffle(imgs)
            imgs = imgs[:max_samples]
        paths.extend(imgs)
        labels.extend([label_idx] * len(imgs))
    return paths, np.array(labels)


class MedicalDataset(Dataset):
    """Simple torch Dataset for CNN baselines."""

    def __init__(self, paths, labels, transform=None):
        self.paths = list(paths)
        self.labels = list(labels)
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                    ),
                ]
            )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
