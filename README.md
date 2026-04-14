# 🎵 MusicGen: AI-Driven BGM Generation

**MusicGen** is a state-of-the-art audio generation pipeline designed to create high-quality background music (BGM) tailored specifically to a user's vocal input (singing or speaking). By leveraging neural audio tokenization and an autoregressive Transformer architecture, the system learns the rhythmic and harmonic relationship between vocals and instrumentation.

---

## 🚀 Key Features

- **Neural Audio Tokenization**: Uses **Meta's EnCodec** to compress raw 24kHz audio into discrete semantic tokens, allowing the Transformer to "read" and "write" music like text.
- **Transformer-Based Generation**: An Encoder-Decoder Transformer architecture that conditions BGM generation on vocal features.
- **Advanced Preprocessing**: 
  - **Noise Gating**: Automatic removal of floor noise below -40dB.
  - **VAD (Voice Activity Detection)**: Precise silence trimming to focus on active singing/speech.
  - **Pitch Analysis**: Extraction of fundamental frequencies (f0) to guide musical accompaniment.
- **Pro Mixing Engine**: Automatically blends vocals with generated BGM using configurable gain stages and a -1.0dB safety headroom to prevent digital clipping.
- **GPU Optimized**: Built to run on consumer hardware (specifically tested on NVIDIA RTX 3050) using **Mixed Precision (FP16)** to maximize VRAM efficiency.

---

## 🏗️ Architecture Overview

### 1. Preprocessing (`core/preprocessor.py`)
Standardizes input audio to 16kHz mono, applies noise reduction, and extracts STFT and pitch features.

### 2. Tokenization (`core/tokenizer.py`)
The bridge between waveforms and AI. It converts continuous audio into a sequence of integers (tokens) from a vocabulary of 1024, enabling the Transformer to handle complex musical structures.

### 3. Generative Model (`core/transformer_model.py`)
An autoregressive Transformer that predicts the next "musical word" (BGM token) based on the "sentence" provided by the vocals.

### 4. Training Pipeline (`train.py`)
A robust training loop featuring:
- **Automatic Checkpointing**: Saves `recent_model.pth` every epoch and preserves the `best_model.pth` based on loss.
- **Mixed Precision**: Uses `torch.amp` to fit large models into 4GB VRAM.

---

## 🛠️ Installation

### 1. Clone & Setup Environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install librosa soundfile numpy tomli encodec einops
```

### 2. Configure System
Edit `config.toml` to adjust audio parameters, mixing gains, or gain thresholds.

---

## 📖 Usage

### Running Inference
To generate BGM for a custom vocal track:
```powershell
.\.venv\Scripts\python.exe main.py --input data/inp/your_vocals.wav --output data/output/final_mix.wav
```

### Training the Model
To train on your own dataset (designed for MUSDB18):
```powershell
.\.venv\Scripts\python.exe train.py
```

---

## 📊 Dataset Requirements
The model is designed for the **MUSDB18-HQ** dataset. 
1. Download from [Zenodo](https://zenodo.org/record/3338373).
2. Place isolated stems in `data/inp/`.
3. The training script will automatically handle vocal isolation and BGM mixing for ground-truth comparison.

---

## 🔧 Technology Stack
- **Language**: Python 3.9+
- **Deep Learning**: PyTorch + CUDA
- **Audio Processing**: Librosa, Soundfile, EnCodec
- **Hardware Target**: NVIDIA RTX 3050 (Laptop/Desktop)

---

**Author:** Ankit Ghosh  
**Project:** Comprehensive BGM Generation via Audio Language Modeling
