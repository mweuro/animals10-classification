import os
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataloaders import train_loader, val_loader
from src.evaluate import evaluating_step
from src.models import ResNet18Model
from src.train import training_step
from src.utils import EarlyStopping, log_epoch




def main(model: nn.Module, 
         train_loader: torch.utils.data.DataLoader, 
         val_loader: torch.utils.data.DataLoader, 
         optimizer: torch.optim.Optimizer, 
         criterion: nn.Module, 
         device: torch.device, 
         epochs: int):
    

    early_stopper = EarlyStopping(patience=5)
    best_val_loss = float("inf")
    csv_path = "logs/training_log.csv"

    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=2
    )

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = training_step(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluating_step(model, val_loader, criterion, device)
        log_epoch(csv_path, epoch, train_loss, train_acc, val_loss, val_acc)

        # Scheduler
        scheduler.step(val_loss)

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/resnet18.pth")
            print("Saved best model!")

        # Early stopping
        early_stopper(val_loss)
        if early_stopper.early_stop:
            break


if __name__ == "__main__":
    model = ResNet18Model(num_classes=10, pretrained=True)
    model.finetune()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.get_trainable_params(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    epochs = 20

    main(model, train_loader, val_loader, optimizer, criterion, device, epochs)