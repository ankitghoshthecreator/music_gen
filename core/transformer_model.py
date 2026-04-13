import torch
import torch.nn as nn

class MusicTransformer(nn.Module):
    """
    Encoder-Decoder Transformer to autoregressively predict BGM from Vocals.
    """
    def __init__(self, vocab_size=1024, num_quantizers=8, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_quantizers = num_quantizers
        self.d_model = d_model
        
        # Summed embedding across codebooks
        self.token_emb = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for _ in range(num_quantizers)
        ])
        
        self.pos_emb = nn.Parameter(torch.zeros(1, 2000, d_model))
        
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        
        # Parallel output heads for each quantizer
        self.out_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(num_quantizers)
        ])

    def _embed(self, x):
        b, q, s = x.shape
        emb = sum(self.token_emb[i](x[:, i, :]) for i in range(q))
        emb = emb + self.pos_emb[:, :s, :]
        return emb

    def forward(self, vocals, bgm):
        """
        vocals & bgm shapes: (Batch, Num_Quantizers, Seq_Len)
        """
        src = self._embed(vocals)
        tgt = self._embed(bgm)
        
        # Causal mask for decoder
        seq_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(vocals.device)
        
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        
        # Predict probability distribution for vocab (batch, num_quantizers, seq_len, vocab_size)
        logits = torch.stack([head(out) for head in self.out_heads], dim=1) 
        return logits
