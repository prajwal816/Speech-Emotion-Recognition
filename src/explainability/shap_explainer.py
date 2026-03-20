import torch
import shap
import numpy as np

class ModelExplainer:
    def __init__(self, model, background_data, device="cuda"):
        """
        model: trained PyTorch model
        background_data: a batch of background examples (e.g., from train set) 
                         tensor of shape (batch_size, 1, n_mels, time)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.background_data = background_data.to(self.device)
        
        # GradientExplainer for deep learning models in PyTorch
        self.explainer = shap.GradientExplainer(self.model, self.background_data)
        
    def explain(self, test_data, target_class=None):
        """
        test_data: tensor of shape (batch, 1, n_mels, time)
        target_class: int representing class index to explain (if None, explains all classes)
        Returns SHAP values.
        """
        test_data = test_data.to(self.device)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(test_data)
        
        if target_class is not None:
            # GradientExplainer returns a list of arrays (one for each class)
            # or a single array depending on the model output.
            if isinstance(shap_values, list):
                return shap_values[target_class]
            return shap_values
            
        return shap_values
