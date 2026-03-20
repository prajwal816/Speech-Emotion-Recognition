import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from src.evaluation.metrics import calculate_metrics, plot_confusion_matrix

class Trainer:
    def __init__(self, model, config, train_loader, val_loader, device="cuda"):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        
        lr = float(config["training"]["learning_rate"])
        wd = float(config["training"]["weight_decay"])
        
        self.optimizer = Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=wd
        )
        
        self.epochs = config["training"]["epochs"]
        self.patience = config["training"]["patience"]
        self.checkpoint_dir = config["training"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc="Training")
        for features, labels in pbar:
            features, labels = features.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * features.size(0)
            pbar.set_postfix({"batch_loss": loss.item()})
            
        return total_loss / len(self.train_loader.dataset)

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * features.size(0)
                
                probs = torch.softmax(logits, dim=1)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        metrics["loss"] = total_loss / len(self.val_loader.dataset)
        return metrics

    def train(self):
        print(f"Starting training on {self.device}")
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            
            print(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_metrics['loss']:.4f} - Val Acc: {val_metrics['accuracy']:.4f} - "
                  f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "best_model.pt"))
                print("Saved new best model.")
            else:
                self.early_stop_counter += 1
                
            if self.early_stop_counter >= self.patience:
                print("Early stopping triggered due to no improvement!")
                break
                
        print("Training complete.")
