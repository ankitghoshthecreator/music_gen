import torch
import librosa
import numpy as np
import argparse
import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.tokenizer import AudioTokenizer
from core.transformer_model import MusicTransformer
from utils.config_loader import load_config
from utils.audio_io import save_audio
from core.mixer import TrackMixer

def main():
    parser = argparse.ArgumentParser(description="Test MusicGen with an MP3 vocal track")
    parser.add_argument("--input", type=str, required=True, help="Path to input MP3 file")
    parser.add_argument("--output", type=str, default="data/output/test_result.wav", help="Output path")
    parser.add_argument("--length", type=float, default=10.0, help="Max length in seconds to process")
    args = parser.parse_args()

    # Create directories if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    config = load_config("config.toml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using {device}")

    # 1. Initialize Tokenizer & Model
    tokenizer = AudioTokenizer(device=device)
    model = MusicTransformer(
        vocab_size=1024, 
        num_quantizers=8, 
        d_model=256, 
        nhead=8, 
        num_layers=4
    ).to(device)

    # 2. Load Weights
    checkpoint_path = "data/checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found at {checkpoint_path}")
        print("Please ensure you have run training and saved 'data/checkpoints/best_model.pth'")
        return
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print("[Model] Weights loaded successfully.")
    except Exception as e:
        print(f"[Error] Failed to load weights: {e}")
        return

    # 3. Load MP3 Vocal
    print(f"[Audio] Loading {args.input}...")
    try:
        y_vocal, sr = librosa.load(args.input, sr=tokenizer.sample_rate, mono=True)
    except Exception as e:
        print(f"[Error] Could not load audio file: {e}")
        return
    
    # Pre-process: Limit length for testing
    if len(y_vocal) > args.length * sr:
        print(f"[Audio] Truncating to {args.length} seconds for faster inference.")
        y_vocal = y_vocal[:int(args.length * sr)]

    # 4. Tokenize Vocals
    print("[Tokenizer] Converting vocals to neural tokens...")
    vocal_tensor = torch.from_numpy(y_vocal).unsqueeze(0).to(device)
    # Tokenize (returns tokens in [1, 8, T])
    vocal_tokens = tokenizer.encode(vocal_tensor, tokenizer.sample_rate)

    # 5. Greedy Inference Loop
    seq_len = vocal_tokens.shape[-1]
    print(f"[Inference] Generating BGM tokens (Sequence length: {seq_len})...")
    
    # Initialize BGM tokens with zeros (placeholder)
    # In a real autoregressive model, we start with a <START> token, 
    # but here we'll grow the sequence from silence tokens.
    bgm_tokens = torch.zeros((1, 8, seq_len), dtype=torch.long, device=device)
    
    # Simple Greedy Decoding
    # We predict bgm_tokens[:, :, i] based on bgm_tokens[:, :, :i]
    with torch.no_grad():
        for i in range(1, seq_len):
            # Pass prefix to model
            # src: vocals, tgt: current bgm prefix
            logits = model(vocal_tokens, bgm_tokens[:, :, :i]) # Output: (1, 8, i, 1024)
            
            # Predict next token at index i
            # The last logit models the token at the NEXT position
            next_tokens = torch.argmax(logits[:, :, -1, :], dim=-1) # (1, 8)
            bgm_tokens[:, :, i] = next_tokens
            
            if i % 10 == 0 or i == seq_len - 1:
                progress = (i / seq_len) * 100
                print(f"  Progress: {progress:.1f}% ({i}/{seq_len} frames)", end="\r")

    print("\n[Inference] Generation complete.")

    # 6. Decode BGM
    print("[Tokenizer] Decoding BGM tokens back to audio...")
    y_bgm = tokenizer.decode(bgm_tokens).cpu().numpy()
    
    # Scale audio if needed (tokenizer.decode might return batch dim)
    if y_bgm.ndim > 1:
        y_bgm = y_bgm.flatten()
    
    # Ensure same length
    y_bgm = y_bgm[:len(y_vocal)]

    # 7. Mix and Save
    print("[Mixer] Creating final mix...")
    mixer = TrackMixer(config)
    y_mix = mixer.mix(y_vocal, y_bgm)
    
    save_audio(args.output, y_mix, sr=tokenizer.sample_rate)
    print(f"--- 🎵 Process Complete 🎵 ---")
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
