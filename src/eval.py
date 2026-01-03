import torch
from torch.utils.data import DataLoader
from dataset import get_dataloaders
from model import Model
from utils import load_config

def evaluate_model(model, dataloader: DataLoader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


if __name__ == "__main__":
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    _, test_loader, classes = get_dataloaders()

    # Recreate model
    model = Model(
        num_classes=len(classes),
        pretrained=False,
        freeze_backbone=False
    )

    # Load weights
    model.load_state_dict(
        torch.load("model.pth", map_location=device)
    )
    model.to(device)

    # Evaluate
    accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")
