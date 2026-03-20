import torch
import torchaudio
import torchaudio.transforms as T
import yaml

class AudioFeatureExtractor:
    def __init__(self, config_path="configs/config.yaml", config_dict=None):
        if config_dict is None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)["data"]
        else:
            self.config = config_dict["data"]
            
        self.sample_rate = self.config["sample_rate"]
        self.n_mels = self.config["n_mels"]
        self.n_mfcc = self.config["n_mfcc"]
        self.n_fft = self.config["n_fft"]
        self.hop_length = self.config["hop_length"]
        
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        self.mfcc = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "n_mels": self.n_mels,
                "hop_length": self.hop_length,
            }
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extracts Mel spectrogram from waveform.
        Args:
            waveform (torch.Tensor): Audio waveform of shape (channels, time)
        Returns:
            torch.Tensor: Mel spectrogram in dB scale
        """
        mel_spec = self.mel_spectrogram(waveform)
        return self.amplitude_to_db(mel_spec)

    def extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extracts MFCC features from waveform.
        Args:
            waveform (torch.Tensor): Audio waveform of shape (channels, time)
        Returns:
            torch.Tensor: MFCC features
        """
        return self.mfcc(waveform)

    def extract_combined_features(self, waveform: torch.Tensor) -> dict:
        """
        Extracts both Mel and MFCC features.
        """
        return {
            "mel": self.extract_mel_spectrogram(waveform),
            "mfcc": self.extract_mfcc(waveform)
        }
