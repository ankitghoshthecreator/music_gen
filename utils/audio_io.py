import librosa
import soundfile as sf
import os
import numpy as np

def load_audio(filepath, target_sr=16000, mono=True):
    """
    Loads an audio file, automatically resampling to target_sr and downmixing to mono.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file {filepath} not found.")
        
    y, sr = librosa.load(filepath, sr=target_sr, mono=mono)
    return y, sr

def save_audio(filepath, y, sr=16000):
    """
    Saves a numpy audio array to a standard WAV file.
    Creates parent directories if they don't exist.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Optional: basic clipping prevention before saving
    y_clipped = np.clip(y, -1.0, 1.0)
    
    sf.write(filepath, y_clipped, sr)
