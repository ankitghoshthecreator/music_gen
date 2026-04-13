import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from core.dataset import DummyMUSDBDataset
from core.transformer_model import MusicTransformer
import os

def main():
    print("Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up tiny hyperparams for initial testing on RTX 3050 (4GB VRAM)
    batch_size = 2
    epochs = 1
    
    print("Initializing dummy dataset...")
    dataset = DummyMUSDBDataset(num_samples=10, seq_len=50) # Small for quick test
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("Initializing Transformer Model (Small Size)...")
    model = MusicTransformer(vocab_size=1024, num_quantizers=8, d_model=128, nhead=4, num_layers=2)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Starting Training Loop...")
    model.train()
    
    # Optional mixed-precision scaler
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (vocals, bgm) in enumerate(dataloader):
            vocals = vocals.to(device)
            bgm = bgm.to(device)
            
            # Autoregressive setup: 
            # Predict the next token target: [1:] by feeding the input [:-1]
            decoder_input = bgm[:, :, :-1]
            target = bgm[:, :, 1:]
            vocals_context = vocals[:, :, :-1]
            
            optimizer.zero_grad()
            
            # Using Mixed Precision (fp16) to prevent OOM errors on smaller GPUs
            with torch.cuda.amp.autocast():
                logits = model(vocals_context, decoder_input)
                
                # Reshape for Loss computation
                logits = logits.reshape(-1, logits.size(-1))
                target = target.reshape(-1)
                
                loss = criterion(logits, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            print(f"Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
            
        epoch_avg_loss = epoch_loss / len(dataloader)
        print(f"--- Average Loss: {epoch_avg_loss:.4f} ---")
        
        os.makedirs("data/checkpoints", exist_ok=True)
        
        # 1. Always save the most recent epoch
        torch.save(model.state_dict(), "data/checkpoints/recent_model.pth")
        
        # 2. Save the best model
        if not hasattr(model, 'best_loss'):
            model.best_loss = float('inf')
            
        if epoch_avg_loss < model.best_loss:
            model.best_loss = epoch_avg_loss
            torch.save(model.state_dict(), "data/checkpoints/best_model.pth")
            print(f"[*] New best model saved with loss: {epoch_avg_loss:.4f}")
        
    print("Training complete! Checkpoints saved to data/checkpoints/")

if __name__ == "__main__":
    main()
