import model
import torch

myModel = model.EfficientNet(3, 2, False)

print("effnet loaded")

with torch.no_grad():
    x = torch.zeros(1, 3, 224, 224)
    out = myModel(x)
    print(f"my effnet shape: {out.shape}")

myModel = model.ResNet_50_224(3, 2, False)

print('resnet loaded')

with torch.no_grad():
    x = torch.zeros(1, 3, 224, 224)
    out = myModel(x)
    print(f"my resnet shape: {out.shape}")




