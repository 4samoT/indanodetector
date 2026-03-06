import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from data.data_loading import TransistorDataset
from models.resnet18 import ResNet18Binary


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
EPOCHS = 8
BATCH_SIZE = 16
LR = 1e-4

MEAN = [0.3865, 0.2763, 0.2414]
STD  = [0.2055, 0.1428, 0.1165]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_ROOT = "datasets/transistor"


# ---------------------------------------------------------
# AUGMENTATIONS
# ---------------------------------------------------------
def get_transforms(split):
    """
    - Mild rotation & perspective: simulates imperfect camera angle
    - Brightness/contrast jitter: simulates lighting variance
    - Blur & noise: simulates slight motion blur on conveyor
    """
    if split == "train":
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomApply([
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            ], p=0.7),
            T.RandomRotation(degrees=5),
            T.RandomPerspective(distortion_scale=0.1, p=0.4),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

# ---------------------------------------------------------
# TRAIN LOOP
# ---------------------------------------------------------
def compute_accuracy(logits, labels):
    preds = (torch.sigmoid(logits) > 0.5).long()
    correct = (preds == labels.long()).sum().item()
    return correct, labels.size(0)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0
    correct_total = 0
    total_samples = 0

    start_time = time.time()
    total_batches = len(loader)

    for batch_idx, (imgs, labels) in enumerate(loader):
        batch_start = time.time()

        imgs = imgs.to(DEVICE)
        labels = labels.float().to(DEVICE)

        logits = model(imgs).squeeze(1)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct, total = compute_accuracy(logits, labels)
        correct_total += correct
        total_samples += total

        # Batch progress
        print(f"  Batch {batch_idx+1}/{total_batches} "
              f"| loss={loss.item():.4f} "
              f"| time={time.time() - batch_start:.3f}s")

    accuracy = correct_total / total_samples
    epoch_time = time.time() - start_time
    return running_loss / total_batches, accuracy, epoch_time


# ---------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------

print(f"Using device: {DEVICE}")

transform_train = get_transforms("train")
transform_test = get_transforms("test")

print("Loading dataset...")
train_set = TransistorDataset(DATASET_ROOT, split="train", transform=transform_train)
test_set = TransistorDataset(DATASET_ROOT, split="test", transform=transform_test)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

print(f"Training samples: {len(train_set)}")
print(f"Testing samples: {len(test_set)}")

model = ResNet18Binary().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    print(f"\n======== EPOCH {epoch+1}/{EPOCHS} ========")

    loss, accuracy, epoch_time = train_epoch(model, train_loader, criterion, optimizer)

    print(f"Epoch {epoch+1} completed")
    print(f"  Avg Loss: {loss:.4f}")
    print(f"  Train Accuracy: {accuracy:.3f}")
    print(f"  Epoch Time: {epoch_time:.2f}s")

torch.save(model.state_dict(), "simple_model.pth")
print("\nModel saved to simple_model.pth")
