import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from core.dataset import MUSDBDataset
from core.transformer_model import MusicTransformer
from core.tokenizer import AudioTokenizer
from utils.config_loader import load_config

def main():
    print("--- 🎵 MusicGen Training Pipeline 🎵 ---")
    
    # 1. Load Configuration
    config = load_config("config.toml")
    train_cfg = config.get("training", {})
    ds_cfg = config.get("dataset", {})
    audio_cfg = config.get("audio", {})
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Using device: {device}")

    # 2. Initialize Tokenizer (shared by dataset)
    # Note: On Windows, to avoid multiprocessing issues with CUDA in DataLoader, 
    # we keep num_workers=0 or ensure the tokenizer is used carefully.
    tokenizer = AudioTokenizer(device=device)
    
    # 3. Initialize Real Datasets
    print(f"[Dataset] Loading training data from {ds_cfg.get('train_path')}...")
    train_dataset = MUSDBDataset(
        root_dir=ds_cfg.get("train_path"),
        tokenizer=tokenizer,
        target_sr=tokenizer.sample_rate,
        segment_duration=train_cfg.get("segment_length_sec", 5.0)
    )

    print(f"[Dataset] Loading validation data from {ds_cfg.get('test_path')}...")
    val_dataset = MUSDBDataset(
        root_dir=ds_cfg.get("test_path"),
        tokenizer=tokenizer,
        target_sr=tokenizer.sample_rate,
        segment_duration=train_cfg.get("segment_length_sec", 5.0)
    )
    
    # Using num_workers=0 to avoid CUDA initialization issues on Windows
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_cfg.get("batch_size", 2), 
        shuffle=True,
        num_workers=0 
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("batch_size", 2),
        shuffle=False,
        num_workers=0
    )
    
    # 4. Initialize Model
    print("[Model] Initializing MusicTransformer...")
    model = MusicTransformer(
        vocab_size=1024, 
        num_quantizers=8, 
        d_model=256, 
        nhead=8, 
        num_layers=4
    )
    model.to(device)
    
    # 5. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.get("learning_rate", 1e-4))
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision for VRAM efficiency on RTX 3050
    scaler = torch.cuda.amp.GradScaler()

    # 6. Training Loop
    epochs = train_cfg.get("epochs", 50)
    checkpoint_dir = train_cfg.get("checkpoint_dir", "data/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"[Training] Starting for {epochs} epochs...")
    best_loss = float('inf')

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        
        for batch_idx, (vocals, bgm) in enumerate(train_loader):
            vocals = vocals.to(device)
            bgm = bgm.to(device)
            
            decoder_input = bgm[:, :, :-1]
            target = bgm[:, :, 1:]
            vocals_context = vocals[:, :, :-1]
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits = model(vocals_context, decoder_input)
                logits = logits.permute(0, 1, 3, 2).contiguous()
                loss = criterion(logits.view(-1, 1024), target.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{epochs}] | Train Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        print(f"[Validation] Running evaluation for epoch {epoch+1}...")
        
        with torch.no_grad():
            for batch_idx, (vocals, bgm) in enumerate(val_loader):
                vocals = vocals.to(device)
                bgm = bgm.to(device)
                
                decoder_input = bgm[:, :, :-1]
                target = bgm[:, :, 1:]
                vocals_context = vocals[:, :, :-1]
                
                with torch.cuda.amp.autocast():
                    logits = model(vocals_context, decoder_input)
                    logits = logits.permute(0, 1, 3, 2).contiguous()
                    loss = criterion(logits.view(-1, 1024), target.reshape(-1))
                
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"    Average Train Loss: {avg_train_loss:.4f}")
        print(f"    Average Val Loss:   {avg_val_loss:.4f}")
        
        # Checkpointing
        if (epoch + 1) % train_cfg.get("save_interval", 1) == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "recent_model.pth"))
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
                print(f"[*] New best model saved (Val Loss: {avg_val_loss:.4f})")

    print(f"Training Complete! Checkpoints saved in {checkpoint_dir}")

if __name__ == "__main__":
    main()
