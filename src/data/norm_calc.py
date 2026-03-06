from pathlib import Path
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from data_loading import TransistorDataset  

# Builds absolute dataset path relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]   
DATASET_ROOT = PROJECT_ROOT / "datasets" / "transistor"

print("Project root:", PROJECT_ROOT)
print("Dataset root:", DATASET_ROOT)

def compute_mean_std(dataset_root):
    tf = transforms.ToTensor()
    ds = TransistorDataset(dataset_root, split="train", transform=tf)

    print("Training samples found:", len(ds))
    if len(ds) == 0:
        raise RuntimeError("No training images found. Check dataset path.")

    loader = DataLoader(ds, batch_size=32, shuffle=False)

    mean = 0.
    std = 0.
    total = 0

    for imgs, _ in loader:
        b = imgs.size(0)
        imgs = imgs.view(b, 3, -1)

        mean += imgs.mean(dim=2).sum(dim=0)
        std += imgs.std(dim=2).sum(dim=0)
        total += b

    mean /= total
    std /= total
    return mean, std


mean, std = compute_mean_std(DATASET_ROOT)
print("Computed mean:", mean)
print("Computed std:", std)
