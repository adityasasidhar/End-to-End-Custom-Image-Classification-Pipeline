import torch
import torchvision.transforms as transforms
from PIL import Image
from dataset import get_dataloaders
from model import Model
from utils import load_config

cfg = load_config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, classes = get_dataloaders()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    ),
])

def load_model(weights_path):
    model = Model(num_classes=len(classes),
                  pretrained=False,
                  freeze_backbone=False
                  )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        conf, probs = torch.max(probs, 1)

    return classes[probs], conf.item()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python infer.py <image_path>")
        exit(1)

    image_path = sys.argv[1]

    model = load_model("model.pth")
    label, confidence = predict(image_path, model)

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")




