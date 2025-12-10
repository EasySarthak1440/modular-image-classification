import torch
from data_module import get_dataloaders
from model_module import get_model
from callback_module import LRA
from train_module import train_model
import torch.optim as optim
import torch.nn as nn

def main():
    train_loader, val_loader = get_dataloaders(batch_size=64)
    model = get_model(freeze=False)  # Set True to freeze conv layers
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    callback = LRA(optimizer, patience=2, stop_patience=3, factor=0.5, threshold=0.75, dwell=True)
    
    train_model(model, train_loader, val_loader, optimizer, criterion, epochs=20,
                callback=callback, ask_epoch=5, device='cuda' if torch.cuda.is_available() else 'cpu',
                log_csv="training_metrics.csv")

if __name__ == "__main__":
    main()
