from torchvision.models import resnet50
import torch.nn as nn


class CharClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CharClassifier, self).__init__()
        self.resnet50 = resnet50()
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)
