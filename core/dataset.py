import torch
from torch.utils.data import Dataset

class DummyMUSDBDataset(Dataset):
    """
    A temporary dummy dataset producing random token sequences arrays.
    Simulates tokenized MUSDB18 data so we can test the Transformer 
    without spending hours downloading 10GB of audio.
    """
    def __init__(self, num_samples=100, seq_len=150, num_quantizers=8, vocab_size=1024):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_quantizers = num_quantizers
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate vocaltokens: shape (num_quantizers, seq_len)
        vocal_tokens = torch.randint(0, self.vocab_size, (self.num_quantizers, self.seq_len), dtype=torch.long)
        # Simulate bgm tokens
        bgm_tokens = torch.randint(0, self.vocab_size, (self.num_quantizers, self.seq_len), dtype=torch.long)
        return vocal_tokens, bgm_tokens
