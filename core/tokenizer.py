import torch
from encodec import EncodecModel
from encodec.utils import convert_audio

class AudioTokenizer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"[Tokenizer] Loading EnCodec model on {self.device}...")
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(6.0) # 6.0 kbps for discrete tokens
        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = self.model.sample_rate

    @torch.no_grad()
    def encode(self, wav, sr):
        """Converts raw audio tensor to compressed integer tokens."""
        wav = wav.to(self.device)
        wav = convert_audio(wav, sr, self.sample_rate, self.model.channels)
        wav = wav.unsqueeze(0) if wav.dim() == 2 else wav 
        
        encoded_frames = self.model.encode(wav)
        codes = encoded_frames[0][0] # (batch, n_quantizers, frames)
        return codes

    @torch.no_grad()
    def decode(self, codes):
        """Reconstructs audio tensor from integer tokens."""
        codes = codes.to(self.device)
        encoded_frames = [(codes, None)]
        wav = self.model.decode(encoded_frames)
        return wav.squeeze(0)
