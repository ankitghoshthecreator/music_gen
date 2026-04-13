import numpy as np
import librosa

class AudioPreprocessor:
    def __init__(self, config):
        self.config = config
        
    def reduce_noise(self, y, sr):
        """Simplistic noise gate based on noise_floor_db."""
        floor_db = self.config.get("noise_reduction", {}).get("noise_floor_db", -40.0)
        floor_amp = librosa.db_to_amplitude(floor_db)
        # Simple noise gate: zero out samples below threshold
        y_cleaned = np.where(np.abs(y) < floor_amp, 0.0, y)
        return y_cleaned

    def apply_vad(self, y, sr):
        """Trims absolute silence or low energy segments."""
        # Using librosa's trim as a robust fallback for the VAD
        yt, index = librosa.effects.trim(y, top_db=60) 
        return yt
        
    def extract_features(self, y, sr):
        """Extracts pitch and STFT for the BGM model."""
        stft_config = self.config.get("stft", {})
        pitch_config = self.config.get("pitch", {})
        
        fmin = pitch_config.get("fmin_hz", 80.0)
        fmax = pitch_config.get("fmax_hz", 400.0)
        n_fft = stft_config.get("window_size", 1024)
        hop_length = stft_config.get("hop_length", 256)
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=fmin, 
            fmax=fmax, 
            sr=sr
        )
        
        S = librosa.stft(
            y, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=stft_config.get("window_type", "hann")
        )
        
        return {
            "pitch_f0": f0,
            "stft_mag": np.abs(S)
        }

    def process(self, y, sr):
        """Run the full preprocessing pipeline."""
        print("Cleaning audio and extracting features...")
        y_clean = self.reduce_noise(y, sr)
        y_vad = self.apply_vad(y_clean, sr)
        features = self.extract_features(y_vad, sr)
        return y_vad, features
