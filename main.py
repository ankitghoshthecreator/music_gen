import argparse
from pathlib import Path
from utils.config_loader import load_config
from utils.audio_io import load_audio

from core.preprocessor import AudioPreprocessor
from core.harmony_engine import HarmonyEngine
from core.arrangement_engine import ArrangementEngine

def main():
    parser = argparse.ArgumentParser(description="Vocal2BGM: Heuristic MIDI Generation Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input vocal wav file")
    parser.add_argument("--output", type=str, default="data/output/background_music.mid", help="Path to output MIDI file")
    parser.add_argument("--bpm", type=float, required=True, help="Target BPM of the vocal track")
    args = parser.parse_args()

    print(f"Loading configuration...")
    config = load_config()

    print(f"Loading input audio from {args.input}...")
    target_sr = config["audio"]["sample_rate"]
    y_raw, sr = load_audio(args.input, target_sr=target_sr, mono=config["audio"]["mono"])
    
    total_duration = len(y_raw) / sr

    # 1. Preprocess Vocals & Extract Melody
    preprocessor = AudioPreprocessor(config)
    y_vocal, melody_notes = preprocessor.process(y_raw, sr, target_bpm=args.bpm)

    if not melody_notes:
        print("Warning: No melody detected in the audio file!")

    # 2. Harmonize (Generate Chords)
    harmony_engine = HarmonyEngine(target_bpm=args.bpm)
    chords = harmony_engine.generate_chords(melody_notes, total_duration)

    # 3. Arrange Tracks (Map to MIDI)
    arrangement_engine = ArrangementEngine(target_bpm=args.bpm)
    midi_data = arrangement_engine.create_midi(melody_notes, chords)

    # 4. Save MIDI
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving generated MIDI to {output_path}...")
    midi_data.write(str(output_path))
    
    print("Pipeline completed successfully! You can load the MIDI into any DAW.")

if __name__ == "__main__":
    main()
