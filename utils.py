import csv
import torch

def save_metrics_to_csv(file_path, epoch, train_loss, train_acc, val_loss, val_acc):
    header = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
    try:
        with open(file_path, 'r'):
            write_header = False
    except FileNotFoundError:
        write_header = True
    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

def save_best_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Saved best model to {file_path}")
