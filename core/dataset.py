import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset
from utils.audio_io import load_audio

class MUSDBDataset(Dataset):
    """
    A real dataset for MUSDB18-HQ. 
    It loads vocal stems and mixes the other stems (bass, drums, other) to form the BGM.
    Segments are randomly sampled and tokenized on the fly.
    """
    def __init__(self, root_dir, tokenizer, target_sr=24000, segment_duration=5.0):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.target_sr = target_sr
        self.segment_samples = int(segment_duration * target_sr)
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")
            
        # Scan for track directories (subfolders)
        self.tracks = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                       if os.path.isdir(os.path.join(root_dir, d))]
        
        if len(self.tracks) == 0:
            raise RuntimeError(f"No track directories found in {root_dir}")
            
        print(f"[Dataset] Found {len(self.tracks)} tracks in {root_dir}")

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track_path = self.tracks[idx]
        
        # 1. Load Vocal Stem
        vocal_path = os.path.join(track_path, "vocals.wav")
        vocal_y, _ = load_audio(vocal_path, target_sr=self.target_sr)
        
        # 2. Load and Mix BGM Stems (bass, drums, other)
        bgm_stems = ["bass.wav", "drums.wav", "other.wav"]
        bgm_y = None
        
        for stem in bgm_stems:
            stem_path = os.path.join(track_path, stem)
            if not os.path.exists(stem_path):
                continue
                
            y, _ = load_audio(stem_path, target_sr=self.target_sr)
            if bgm_y is None:
                bgm_y = y
            else:
                # Mix by adding (ensure same length)
                min_len = min(len(bgm_y), len(y))
                bgm_y = bgm_y[:min_len] + y[:min_len]
        
        if bgm_y is None:
            # Fallback if no stems found (use mixture but subtract vocals if possible, 
            # or just use mixture as proxy if desperate. Here we just use a zero array).
            bgm_y = np.zeros_like(vocal_y)
            
        # 3. Synchronize lengths
        min_len = min(len(vocal_y), len(bgm_y))
        vocal_y = vocal_y[:min_len]
        bgm_y = bgm_y[:min_len]
        
        # 4. Random Segment Extraction
        if min_len > self.segment_samples:
            start = random.randint(0, min_len - self.segment_samples)
            v_seg = vocal_y[start : start + self.segment_samples]
            b_seg = bgm_y[start : start + self.segment_samples]
        else:
            # Pad if shorter than target segment duration
            v_seg = np.pad(vocal_y, (0, self.segment_samples - min_len))
            b_seg = np.pad(bgm_y, (0, self.segment_samples - min_len))
            
        # 5. Tokenization
        # Convert to torch tensors (1, samples)
        v_tensor = torch.from_numpy(v_seg).float().unsqueeze(0)
        b_tensor = torch.from_numpy(b_seg).float().unsqueeze(0)
        
        # EnCodec expects 1D or 2D (batch, samples)
        # Tokenizer returns (1, n_quantizers, seq_len)
        try:
            v_tokens = self.tokenizer.encode(v_tensor, self.target_sr)
            b_tokens = self.tokenizer.encode(b_tensor, self.target_sr)
            
            # Remove batch dimension for DataLoader
            return v_tokens.squeeze(0), b_tokens.squeeze(0)
        except Exception as e:
            print(f"Error tokenizing track {track_path}: {e}")
            # Return dummy on error to avoid crashing the whole loop
            return torch.zeros((8, 150), dtype=torch.long), torch.zeros((8, 150), dtype=torch.long)
