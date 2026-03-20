import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true, y_pred_probs):
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    
    try:
        if y_pred_probs.shape[1] > 2:
            roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="macro")
        else:
            roc_auc = roc_auc_score(y_true, y_pred_probs[:, 1])
    except Exception as e:
        roc_auc = float('nan')
        
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": acc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }

def plot_confusion_matrix(cm, classes, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
