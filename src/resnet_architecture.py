from torchvision import models

# Just the architecture, no pre-trained weights
original_resnet = models.resnet18(weights=None)

print(original_resnet)