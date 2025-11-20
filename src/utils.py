import csv
import os



class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss does not improve.
    """
    def __init__(self, patience: int = 5) -> None:
        """
        Initializes the EarlyStopping object.

        Args:
            patience (int): Number of epochs to wait for improvement before stopping. Default is 5.

        Returns:
            None
        """
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
    def __call__(self, val_loss: float) -> None:
        """
        Checks if early stopping condition is met based on the validation loss.

        Args:
            val_loss (float): Current validation loss.

        Returns:
            None
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            print("Early stopping triggered.")


def log_epoch(csv_path: str, 
              epoch: int,
              train_loss: float, 
              train_acc: float,
              val_loss: float, 
              val_acc: float):
    """
    Logs the training and validation metrics for an epoch into a CSV file.  
    
    Args:
        csv_path (str): Path to the CSV file.
        epoch (int): Current epoch number.
        train_loss (float): Training loss for the epoch.
        train_acc (float): Training accuracy for the epoch.
        val_loss (float): Validation loss for the epoch.
        val_acc (float): Validation accuracy for the epoch.

    Returns:
        None
    """
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "epoch",
                "train_loss", "train_acc",
                "val_loss", "val_acc"
            ])

        writer.writerow([
            epoch,
            train_loss, train_acc,
            val_loss, val_acc,
        ])
