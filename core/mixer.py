import numpy as np
import librosa

class TrackMixer:
    def __init__(self, config):
        self.config = config
        
    def mix(self, vocal_y, bgm_y):
        """Mixes tracks together applying gain and headroom limits."""
        print("Mixing vocals and generated BGM...")
        mix_config = self.config.get("mix", {})
        
        vocal_gain = mix_config.get("vocal_gain", 1.0)
        bgm_gain = mix_config.get("bgm_gain", 0.6)
        headroom_db = mix_config.get("headroom_db", -1.0)
        
        # Ensure array lengths match exactly by padding the shorter one
        max_len = max(len(vocal_y), len(bgm_y))
        v_padded = np.pad(vocal_y, (0, max_len - len(vocal_y)))
        b_padded = np.pad(bgm_y, (0, max_len - len(bgm_y)))
        
        # Mix with configured gains
        mixed = (v_padded * vocal_gain) + (b_padded * bgm_gain)
        
        # Apply headroom normalization
        headroom_amp = librosa.db_to_amplitude(headroom_db)
        max_peak = np.max(np.abs(mixed))
        if max_peak > 0: # Avoid div by zero
            # Normalize to headroom if it clips or exceeds headroom
            if max_peak > headroom_amp:
                 mixed = mixed * (headroom_amp / max_peak)
                 
        return mixed
