import torch
import copy
from utils import save_best_model

class LRA:
    def __init__(self, optimizer, patience=2, stop_patience=3, factor=0.5, threshold=0.8,
                 dwell=True, model_name="model", save_path="best_model.pth"):
        self.optimizer = optimizer
        self.patience = patience
        self.stop_patience = stop_patience
        self.factor = factor
        self.threshold = threshold
        self.dwell = dwell
        self.model_name = model_name
        self.save_path = save_path
        
        self.best_loss = float('inf')
        self.best_model_weights = None
        self.epochs_no_improve = 0
        self.lr_adjustments = 0
    
    def step(self, val_loss, model, train_acc):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            if self.dwell:
                self.best_model_weights = copy.deepcopy(model.state_dict())
            save_best_model(model, self.save_path)
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        
        # Check for learning rate reduction
        if train_acc > self.threshold and self.epochs_no_improve >= self.patience:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.factor
            self.epochs_no_improve = 0
            self.lr_adjustments += 1
            print(f"Reduced LR to {self.optimizer.param_groups[0]['lr']}")
            
            if self.dwell and self.best_model_weights is not None:
                model.load_state_dict(self.best_model_weights)
                print("Restored best model weights.")
        
        # Stop training after consecutive LR reductions without improvement
        if self.lr_adjustments >= self.stop_patience:
            print("Early stopping triggered.")
            return True  # Stop training
        return False
