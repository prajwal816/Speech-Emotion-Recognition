import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import shap

def plot_mel_spectrogram(mel_spec, original_sr=16000, hop_length=512, title="Mel Spectrogram", save_path=None):
    """
    Plots a Mel spectrogram.
    mel_spec: numpy array of shape (n_mels, time)
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec, sr=original_sr, hop_length=hop_length, 
                             x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_shap_summary(shap_values, feature_names=None, save_path=None):
    """
    shap_values: numpy array of shape (samples, features) (flattened SHAP values for summary)
    """
    plt.figure()
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_shap_spectrogram(original_mel, shap_matrix, save_path=None):
    """
    Plots the original spectrogram alongside its SHAP attribution map.
    original_mel: shape (n_mels, time)
    shap_matrix: shape (n_mels, time)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Original
    img1 = librosa.display.specshow(original_mel, x_axis='time', y_axis='mel', ax=axes[0])
    axes[0].set_title("Original Mel Spectrogram")
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
    
    # SHAP Feature Attribution
    vmax = np.max(np.abs(shap_matrix))
    if vmax == 0:
        vmax = 1e-5
        
    img2 = axes[1].imshow(shap_matrix, aspect='auto', origin='lower', cmap='RdBu_r', 
                          vmin=-vmax, vmax=vmax)
    axes[1].set_title("SHAP Feature Attribution")
    axes[1].set_xlabel("Time Frames")
    axes[1].set_ylabel("Mel Frequency Bins")
    fig.colorbar(img2, ax=axes[1], label='SHAP Value')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
