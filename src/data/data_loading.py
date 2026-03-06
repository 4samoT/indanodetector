from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class TransistorDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split
        self.samples = []
        self.labels = []

        if split == "train":
            good = self.root / "train" / "good"
            imgs = list(good.glob("*.png"))
            self.samples.extend(imgs)
            self.labels.extend([0] * len(imgs))

        else:
            test_root = self.root / "test"
            for cls_dir in test_root.iterdir():
                if cls_dir.is_dir():
                    imgs = list(cls_dir.glob("*.png"))
                    label = 0 if cls_dir.name == "good" else 1
                    self.samples.extend(imgs)
                    self.labels.extend([label] * len(imgs))

        self.transform = transform or T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label
