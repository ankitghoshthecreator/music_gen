# Vocal2BGM 🎵  
### AI-Powered Background Music Generation from Vocal Input using MIDI

## Overview

**Vocal2BGM** is an intelligent music generation system that creates background instrumental music when a user uploads a **vocal-only audio file** (acapella, humming, rap vocals, melody singing, etc.).

Instead of directly generating raw audio (which is computationally expensive and unstable), this project uses a **symbolic music generation pipeline with MIDI**, making it faster, controllable, and practical for solo developers.

---

## Core Idea

Convert:

```text
User Vocal Audio (.wav / .mp3)
        ↓
Analyze Pitch + BPM + Key + Emotion
        ↓
Generate Chords + Bass + Drums + Pads (MIDI)
        ↓
Render MIDI into Real Instrumental Audio
        ↓
Final Background Music Output

Why MIDI Instead of Direct Audio Generation?
Direct Audio Generation Problems
Very high dimensional output
Difficult training
Poor long-term song structure
Noise artifacts
Requires massive datasets + GPUs
MIDI Advantages
Lightweight symbolic representation
Easy chord / rhythm control
Better harmony generation
Fast generation
Professional DAW compatibility
Separate instrument tracks possible
What is MIDI?

MIDI = Musical Instrument Digital Interface

MIDI stores instructions like:

Play C4 at 0.0 sec
Hold 1 sec
Play Kick Drum at beat 1
Play Bass A2 louder

It does not store sound.

Features
Input Support
Vocal singing
Humming
Rap vocals
Melody ideas
Acapella songs
AI Analysis
BPM Detection
Pitch Tracking
Key Detection
Mood Detection
Section Segmentation (verse / chorus)
Music Generation
Chord Progressions
Bassline
Drum Groove
Piano Layer
Strings / Pads
Counter Melody
Output Options
Full instrumental .wav
MIDI file .mid
Individual stems
Multiple genre versions
## Tech Stack

- **Audio Processing**: Python, `librosa`, `numpy`
- **MIDI Processing**: `pretty_midi`

## Proposed Architecture

### Stage 1: Robust Input & Preprocessing
- **Action**: The system extracts the melody pitch contour using `librosa.pyin`.
- **Processing**: The pitch is discretized into MIDI notes on a grid defined by the provided target BPM.

### Stage 2: Heuristic Harmonization
- **Action**: Analyzes the melody notes inside each musical bar.
- **Workflow**: Assigns a chord (major/minor) using a heuristic that sets the most frequent melody note in that bar as the chord's root.

### Stage 3: Template-Based Arrangement Engine
- **Action**: Maps the generated chords to MIDI tracks.
- **Workflow**: 
  - Generates a Pad track playing the chords.
  - Generates a Bass track playing root notes.
  - Drops a pre-composed standard 4/4 Drum loop.

### Stage 4: MIDI Output
- **Action**: Compiles all tracks (Melody, Pad, Bass, Drums) into a single `.mid` file.
- **Usage**: The generated MIDI file can be dragged and dropped into any professional DAW (FL Studio, Ableton, Logic) to apply high-quality VST instruments.

## Example Workflow

1. User provides: `sad_vocal.wav` and Target BPM = `82`
2. Pipeline extracts the melody as MIDI notes.
3. System assigns chords per bar (e.g., Am -> F -> C -> G).
4. System arranges Bass, Pads, and Drums.
5. Final Output: `background_music.mid`
Folder Structure
Vocal2BGM/
│── data/
│── models/
│── midi/
│── outputs/
│── notebooks/
│── src/
│   │── preprocessor.py
│   │── harmony_engine.py
│   │── arrangement_engine.py
│── app.py
│── requirements.txt
│── README.md
Installation
git clone https://github.com/yourusername/Vocal2BGM.git
cd Vocal2BGM
pip install -r requirements.txt
## Run Project

```bash
python main.py --input path/to/vocal.wav --bpm 120 --output data/output/bgm.mid
```

Upload a vocal file and specify its BPM to receive a generated instrumental MIDI.

Future Enhancements
Genre selector (LoFi / EDM / Cinematic / Trap)
Auto mastering
Voice-aware dynamic mixing
Web application deployment
Real-time generation
Mobile app
Multi-language folk style generation
Use Cases
Singers without producers
YouTube creators
Reel background music
Demo creation
Music students
Independent composers
Challenges
Wrong chord prediction
Off-tempo vocals
Noisy microphone recordings
Generic generated music
Human-level arrangement complexity
Research Value

This project combines:

Digital Signal Processing
Music Information Retrieval
Sequence Modeling
Symbolic AI Music Generation
Audio Rendering Systems
Final Goal

Create a system where anyone can sing an idea and instantly receive studio-style background music.

f