import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataloaders import train_loader, val_loader
from src.evaluate import evaluating_step
from src.models import ResNet18Model
from src.train import training_step
from src.utils import EarlyStopping, log_epoch


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main(model: nn.Module, 
         train_loader: torch.utils.data.DataLoader, 
         val_loader: torch.utils.data.DataLoader, 
         optimizer: torch.optim.Optimizer, 
         criterion: nn.Module, 
         device: torch.device, 
         config: dict):
    """
    Main training loop for the model.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to perform computations on.
        config (dict): Configuration dictionary with training parameters.
    
    Returns:
        None
    """
    epochs = config['TRAINING']['epochs']
    patience = config['TRAINING']['patience']
    reduce_lr_factor = config['TRAINING']['reduce_lr_factor']
    reduce_lr_patience = config['TRAINING']['reduce_lr_patience']

    early_stopper = EarlyStopping(patience=patience)
    best_val_loss = float("inf")
    csv_path = "logs/training_log.csv"

    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=reduce_lr_factor,
        patience=reduce_lr_patience
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
            print(f"Early stopping triggered at epoch {epoch+1}")
            break


if __name__ == "__main__":
    # Load configuration
    config = load_config("config.yaml")
    
    # Model parameters from config
    num_classes = config['TRAINING']['num_classes']
    learning_rate = config['TRAINING']['learning_rate']
    pretrained = config['MODEL']['pretrained']
    epochs = config['TRAINING']['epochs']
    
    # Initialize model
    model = ResNet18Model(num_classes=num_classes, pretrained=pretrained)
    model.finetune()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.get_trainable_params(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Start training
    main(model, train_loader, val_loader, optimizer, criterion, device, config)