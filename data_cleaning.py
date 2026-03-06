from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class MVTecTransistorDataset(Dataset):
    """
    Basic dataset loader for MVTec AD (for now, just the transistor subset).

    Loads:
        - Normal (good) images for training
        - Normal + defective images for testing
    """

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        if split == "train":
            img_dir = self.root_dir / "train" / "good"
            self.samples = list(img_dir.glob("*.png"))
            self.labels = [0] * len(self.samples)  # 0 = normal

        elif split == "test":
            self.samples = []
            self.labels = []

            test_root = self.root_dir / "test"

            for defect_dir in test_root.iterdir():
                if defect_dir.is_dir():
                    imgs = list(defect_dir.glob("*.png"))
                    self.samples.extend(imgs)

                    label = 0 if defect_dir.name == "good" else 1
                    self.labels.extend([label] * len(imgs))

        else:
            raise ValueError(f"Invalid split '{split}'. Choose 'train' or 'test'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label, str(img_path)


def build_transforms(image_size=256, is_train=True):
    """
    Data augmentation pipeline tailored for industrial vision variations.
    """
    if is_train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.RandomPerspective(distortion_scale=0.1, p=0.3),
            T.RandomApply([T.GaussianBlur(3)], p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
        ])


def get_dataloaders(root_dir, batch_size=16, image_size=256, num_workers=4):
    """
    Convenience function for building train/test dataloaders.
    """

    train_tf = build_transforms(image_size=image_size, is_train=True)
    test_tf = build_transforms(image_size=image_size, is_train=False)

    train_ds = MVTecTransistorDataset(root_dir, split="train", transform=train_tf)
    test_ds = MVTecTransistorDataset(root_dir, split="test", transform=test_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


root = "datasets/transistor"

train_loader, test_loader = get_dataloaders(root)

print("Train batches:", len(train_loader))
print("Test batches:", len(test_loader))
