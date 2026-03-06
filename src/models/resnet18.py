import torch.nn as nn
from torchvision.models import resnet18

# ResNet18 binary classifier
class ResNet18Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Linear(512, 1)  

    def forward(self, x):
        return self.backbone(x)

