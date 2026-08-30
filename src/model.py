import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_model(num_classes=10, pretrained=True):
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
