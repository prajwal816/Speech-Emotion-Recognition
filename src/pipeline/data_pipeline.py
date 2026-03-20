import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple

from src.features.audio_features import AudioFeatureExtractor

class NoiseAugmentation:
    def __init__(self, snr_min: int = 0, snr_max: int = 15):
        self.snr_min = snr_min
        self.snr_max = snr_max

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        snr_db = torch.randint(self.snr_min, self.snr_max + 1, (1,)).item()
        
        # Calculate signal power and noise power
        signal_power = waveform.norm(p=2) ** 2 / waveform.numel()
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / max(snr_linear, 1e-8)
        
        # Generate noise
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

def speaker_normalization(feature: torch.Tensor) -> torch.Tensor:
    """Zero-mean unit-variance normalization per utterance/speaker."""
    mean = feature.mean()
    std = feature.std()
    if std > 1e-6:
        return (feature - mean) / std
    return feature - mean

class EmotionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, config: dict, is_train: bool = True):
        self.data = dataframe
        self.config = config
        self.is_train = is_train
        self.feature_extractor = AudioFeatureExtractor(config_dict=config)
        
        self.target_length = int(self.config["data"]["sample_rate"] * self.config["data"]["duration"])
        self.noise_aug = NoiseAugmentation(
            snr_min=self.config["data"]["augmentation"]["noise_snr_min"],
            snr_max=self.config["data"]["augmentation"]["noise_snr_max"]
        )
        self.apply_prob = self.config["data"]["augmentation"]["apply_prob"]
        
    def __len__(self):
        return len(self.data)
        
    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[1] > self.target_length:
            return waveform[:, :self.target_length]
        elif waveform.shape[1] < self.target_length:
            pad_size = self.target_length - waveform.shape[1]
            return torch.nn.functional.pad(waveform, (0, pad_size))
        return waveform
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.data.iloc[idx]
        audio_path = row["path"]
        label = row["label"]
        
        # Load audio if exists, else fallback to dummy
        if os.path.exists(audio_path):
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.config["data"]["sample_rate"]:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.config["data"]["sample_rate"])
                waveform = resampler(waveform)
            if waveform.shape[0] > 1: # Convert stereo to mono
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        else:
            waveform = torch.randn(1, self.target_length)
            
        waveform = self._pad_or_trim(waveform)
        
        # Data Augmentation (Noise)
        if self.is_train and torch.rand(1).item() < self.apply_prob:
            waveform = self.noise_aug(waveform)
            
        # Extract features (Mel Spectrogram is used for modeling)
        features = self.feature_extractor.extract_mel_spectrogram(waveform)
        
        # Speaker/Instance Normalization
        features = speaker_normalization(features)
        
        return features, label

def get_dataloaders(metadata_csv: str, config: dict):
    """
    Returns train, val, test dataloaders containing the stratified split
    """
    if os.path.exists(metadata_csv):
        df = pd.read_csv(metadata_csv)
    else:
        print(f"Warning: {metadata_csv} not found. Using simulated dataset.")
        # Ensure repeatable simulation
        np.random.seed(config["seed"])
        df = pd.DataFrame({
            "path": [f"dummy_{i}.wav" for i in range(1000)],
            "label": np.random.randint(0, config["model"]["num_classes"], 1000),
            "speaker_id": np.random.randint(0, 50, 1000)
        })
        
    # Stratified Split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config["seed"])
    train_idx, temp_idx = next(sss.split(df, df["label"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=config["seed"])
    val_idx, test_idx = next(sss2.split(temp_df, temp_df["label"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    
    train_dataset = EmotionDataset(train_df, config, is_train=True)
    val_dataset = EmotionDataset(val_df, config, is_train=False)
    test_dataset = EmotionDataset(test_df, config, is_train=False)
    
    # Num workers is set to 0 to avoid multiprocessing issues in dummy/windows setup by default, 
    # but uses config value if carefully managed
    num_workers = config["data"].get("num_workers", 0)
    
    train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
