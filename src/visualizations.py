import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_results(log_file: str) -> None:
    """
    Plots training and validation accuracy and loss from a CSV log file.

    Args:
        log_file (str): Path to the CSV log file containing 'epoch', 'train_loss', 
        'val_loss', 'train_acc', and 'val_acc' columns.
    
    Returns:
        None
    """
    # Load the log data
    log_data = pd.read_csv(log_file)

    # Plot training and validation accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(log_data['epoch'], log_data['train_acc'], label='train acc')
    plt.plot(log_data['epoch'], log_data['val_acc'], label='val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over epochs')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(log_data['epoch'], log_data['train_loss'], label='train loss')
    plt.plot(log_data['epoch'], log_data['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()

    if os.path.exists('plots') is False:
        os.makedirs('plots')
    plt.savefig('plots/training_validation_plots.png')


if __name__ == "__main__":
    plot_results('logs/training_log.csv')