import os
import yaml
import torch
import numpy as np

from src.pipeline.data_pipeline import get_dataloaders
from src.models.cnn_lstm import CNNLSTM
from src.training.trainer import Trainer
from src.explainability.shap_explainer import ModelExplainer
from src.explainability.visualizations import plot_shap_spectrogram

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    print("Loading configuration...")
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(config["seed"])
    
    print("Setting up data pipeline...")
    # 'metadata.csv' will be generated if not present
    train_loader, val_loader, test_loader = get_dataloaders("metadata.csv", config)
    
    print("Initializing CNN-LSTM model...")
    model = CNNLSTM(config)
    
    print("Initializing Trainer...")
    trainer = Trainer(model, config, train_loader, val_loader)
    
    print("\n--- Starting Training Pipeline ---")
    trainer.train()
    
    # Explainability Pipeline
    print("\n--- Starting Explainability Analysis (SHAP) ---")
    model.eval()
    
    # Get background data for SHAP
    bg_features, _ = next(iter(train_loader))
    bg_features = bg_features[:10]  # Subset for speed
    
    explainer = ModelExplainer(model, bg_features, device=trainer.device)
    
    # Get a sample
    test_features, test_labels = next(iter(test_loader))
    sample_to_explain = test_features[:1]
    target_class = test_labels[0].item()
    
    print(f"Explaining prediction for class id: {target_class}")
    shap_attr = explainer.explain(sample_to_explain, target_class=target_class)
    
    # Plotting SHAP Spec
    original_mel = sample_to_explain[0, 0].cpu().numpy()
    shap_matrix = shap_attr[0, 0] # (n_mels, time)
    
    os.makedirs("experiments/visualizations", exist_ok=True)
    save_path = "experiments/visualizations/shap_spectrogram.png"
    plot_shap_spectrogram(original_mel, shap_matrix, save_path=save_path)
    print(f"SHAP visualization saved to {save_path}")

    print("\nPipeline Completed Successfully.")

if __name__ == "__main__":
    main()
