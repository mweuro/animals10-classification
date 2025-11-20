import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm



def training_step(model: nn.Module, 
                  train_loader: torch.utils.data.DataLoader,
                  optimizer: optim.Optimizer,
                  criterion: nn.Module,
                  device: torch.device) -> tuple[float, float]:
    """
    Performs a single training step (epoch) for the model.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        optimizer (optim.Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the training on (CPU or GPU).

    Returns:
        tuple[float, float]: Training loss and accuracy for the epoch.
    """
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc="Train")

    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        acc = correct / total
        train_bar.set_postfix(loss=loss.item(), acc=f"{acc:.3f}")

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    print(f"\nTrain loss = {train_loss:.4f}, Train acc = {train_acc:.4f}")
    return train_loss, train_acc
