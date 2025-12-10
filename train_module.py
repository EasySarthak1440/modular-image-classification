import torch
import torch.nn as nn
from utils import save_metrics_to_csv

def train_model(model, train_loader, val_loader, optimizer, criterion,
                epochs=10, callback=None, ask_epoch=5, device='cpu', log_csv="metrics.csv"):
    
    model.to(device)
    
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= total
        val_acc = correct / total
        
        print(f"Epoch [{epoch}/{epochs}] Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f} Val Loss: {val_loss:.4f}")
        
        # Log metrics
        save_metrics_to_csv(log_csv, epoch, train_loss, train_acc, val_loss, val_acc)
        
        # Callback step
        if callback and callback.step(val_loss, model, train_acc):
            break
        
        # Ask user whether to continue after ask_epoch
        if epoch % ask_epoch == 0:
            cont = input(f"Epoch {epoch} completed. Continue training? (y/n): ")
            if cont.lower() != 'y':
                break
