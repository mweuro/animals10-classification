import csv
import os



class EarlyStopping:
    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
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
