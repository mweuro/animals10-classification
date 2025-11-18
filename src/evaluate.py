import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm



def evaluating_step(model: nn.Module,
                    val_loader: torch.utils.data.DataLoader,
                    criterion: nn.Module,
                    device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    running_loss = 0

    val_bar = tqdm(val_loader, desc="Validate")

    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            acc = correct / total
            val_bar.set_postfix(acc=f"{acc:.3f}", loss=loss.item())

    val_acc = correct / total
    val_loss = running_loss / len(val_loader)

    print(f"\nVal loss = {val_loss:.4f}, Val acc = {val_acc:.4f}")
    return val_loss, val_acc

