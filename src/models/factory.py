import torch.nn as nn 
from torchvision import models

class ModelFactory: 
    @staticmethod
    def get_model(name: str, num_classes: int = 10, in_channels: int = 1, pretrained: bool = False):
        weights = 'DEFAULT' if pretrained else None
        name = name.lower()
        
        if name == 'resnet18':
            model = models.resnet18(weights=weights)
            if in_channels != 3:
                model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model 
        
        elif name == 'alexnet':
            model = models.alexnet(weights=weights)
            if in_channels != 3:
                model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=4, stride=1, padding=2)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            return model
        
        elif name == 'vgg11':
            model = models.vgg11(weights=weights)
            if in_channels != 3: 
                model.features[0] = nn.Conv2d(in_channels, 64, kernel_size = 3, padding=1)
            model.features[20] = nn.Identity()
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            return model
        else: 
            raise ValueError(f"model {name} not founded")