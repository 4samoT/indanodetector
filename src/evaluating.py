import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from data.data_loading import TransistorDataset
from models.resnet18 import ResNet18Binary

# Values obtained from src/data/norm_calc.py
MEAN = [0.3865, 0.2763, 0.2414]
STD  = [0.2055, 0.1428, 0.1165]

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

dataset_root = "datasets/transistor"
device = "cuda" if torch.cuda.is_available() else "cpu"

test_set = TransistorDataset(dataset_root, split="test", transform=transform)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

print(f"Testing samples: {len(test_set)}")

model = ResNet18Binary().to(device)
model.load_state_dict(torch.load("simple_model.pth", map_location=device))
model.eval()

y_true, y_pred, y_scores = [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        logits = model(imgs).squeeze(1)
        probs = torch.sigmoid(logits)

        preds = (probs > 0.5).long().cpu()

        y_true.extend(labels)
        y_pred.extend(preds)
        y_scores.extend(probs.cpu())

y_true = [int(x) for x in y_true]
y_pred = [int(x) for x in y_pred]

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["good", "defect"]))

try:
    print("AUROC:", roc_auc_score(y_true, y_scores))
except:
    print("AUROC could not be computed (only 1 class predicted).")
