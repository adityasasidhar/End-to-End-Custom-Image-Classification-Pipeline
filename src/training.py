import torch
import torch.nn as nn
from dataset import get_dataloaders
from model import Model
from utils import load_config, set_seed, get_device

cfg = load_config()
set_seed(cfg["runtime"]["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

train_loader, test_loader, classes = get_dataloaders()

# Model
model = Model(
    num_classes=len(classes),
    pretrained=True,
    freeze_backbone=True
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

# Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.1
)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


best_acc = 0.0
num_epochs = 15

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    val_acc = evaluate(model, test_loader)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "model.pth")

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {avg_loss:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

    if epoch == 4:
        for param in model.model.layer4.parameters():
            param.requires_grad = True

    if epoch == 10:
        for param in model.model.layer3.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4
        )

    scheduler.step()

print("Training complete. Best Val Acc:", best_acc)
