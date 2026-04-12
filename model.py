import torch
import torch.nn as nn
from torchvision import models



class EnsembleModels(nn.Module):
    def __init__(self, backbones):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)

    def extract_features(self, x):
        return [backbone(x) for backbone in self.backbones]
    
    def forward(self, x):
        raise NotImplementedError

class ProjectionEnsemble(EnsembleModels):
    def __init__(self, backbones, feature_dims, hidden_dims = 256):
        super().__init__(backbones)

        total_dims = sum(feature_dims)

        self.head = nn.Sequential(
            nn.Linear(total_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 2)
        )

    def forward(self, x):
        features = self.extract_features(x)
        out = self.head(torch.cat(features, dim=-1))

        return out

class VotingEnsemble(EnsembleModels):
    def __init__(self, backbones, mode='hard'):
        super().__init__(backbones)
        self.mode = mode
    
    def forward(self, x):
        logits = [backbone(x) for backbone in self.backbones]
        stacked = torch.stack(logits, dim=0)  # (num_models, batch, 2)

        if self.mode == 'soft':
            return stacked.mean(dim=0)
        elif self.mode == 'hard':
            votes = stacked.argmax(dim=-1)  # (num_models, batch) — reuse stacked
            return torch.mode(votes, dim=0).values
        else:
            raise NotImplementedError


# =========== FOUNDATION MODELS =============

def ResNet_50_224(in_channels=3, num_classes=2, voting=True):

    model = models.resnet50('IMAGENET1K_V1')

    if in_channels != 3:
        raise ValueError("Input channels must be 3 for pretrained ResNet-50")
    
    if voting: 
        in_ftrs = model.fc.in_features
        # model.fc = nn.Linear(in_ftrs, num_classes, bias=True)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_ftrs, num_classes)
        )
    else:
        in_ftrs = model.fc.in_features
        model.fc = nn.Identity() 
    
        
    return model

def EfficientNet(in_channels=3, num_classes=2, voting=True):
    """
    Creates an EfficientNetB4 model pretrained on ImageNet1k.
    
    Args:
        in_channels (int): Number of input channels (3=RGB, 1=grayscale)
        num_classes (int): Number of output classes
    
    Returns:
        model (torch.nn.Module): EfficientNetV2-L pretrained model adapted for num_classes
        transform (callable): Timm-compatible validation transform for 384x384 images
    """

    if in_channels != 3:
        raise ValueError("Input channels must be 3 for pretrained EfficientNet.")

    model = models.efficientnet_b4(weights='IMAGENET1K_V1')

    if voting:
        in_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(in_ftrs, num_classes, bias=True)
    else:
        in_ftrs = model.classifier[1].in_features
        model.classifier = nn.Identity()

    return model

        


