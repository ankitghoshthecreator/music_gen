import argparse
from pathlib import Path
from utils.config_loader import load_config
from utils.audio_io import load_audio
import subprocess
from pydub import AudioSegment

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
        detected_key = (0, 'major')
    else:
        detected_key = preprocessor.detect_key(melody_notes)
        root_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][detected_key[0]]
        print(f"Detected Key: {root_name} {detected_key[1]}")

    # 2. Harmonize (Generate Chords)
    harmony_engine = HarmonyEngine(target_bpm=args.bpm, config=config)
    chords = harmony_engine.generate_chords(melody_notes, total_duration, detected_key=detected_key)

    # 3. Arrange Tracks (Map to MIDI)
    arrangement_engine = ArrangementEngine(target_bpm=args.bpm, config=config)
    midi_data = arrangement_engine.create_midi(melody_notes, chords)

    # 4. Save MIDI
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving generated MIDI to {output_path}...")
    midi_data.write(str(output_path))
    
    # 5. Synthesize MIDI to Audio
    synth_cfg = config.get("synthesis", {})
    sf2_path = Path(synth_cfg.get("soundfont_path")).absolute()
    bgm_wav_path = Path(synth_cfg.get("output_bgm_wav", "data/output/bgm_only.wav")).absolute()
    
    print(f"Synthesizing BGM using FluidSynth with SoundFont: {sf2_path}...")
    try:
        # Construct the command directly to avoid midi2audio wrapper issues
        # fluidsynth [options] [soundfonts] [midifiles]
        cmd = [
            "fluidsynth",
            "-ni",
            "-F", str(bgm_wav_path),
            "-r", "44100",
            "-g", "1.0",
            str(sf2_path),
            str(output_path.absolute())
        ]
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FluidSynth Error: {result.stderr}")
            # Fallback: maybe the version needs different flags?
            # Some versions use -T wav
    except Exception as e:
        print(f"Error during synthesis: {e}")
        return

    # 6. Mix Vocal and BGM
    mix_cfg = config.get("mix", {})
    vocal_gain = mix_cfg.get("vocal_gain_db", 0.0)
    bgm_gain = mix_cfg.get("bgm_gain_db", -6.0)
    final_output_path = synth_cfg.get("output_final_wav", "data/output/final_prototype.wav")
    
    print(f"Mixing vocals and BGM into {final_output_path}...")
    try:
        vocal_audio = AudioSegment.from_file(args.input)
        bgm_audio = AudioSegment.from_file(bgm_wav_path)
        
        # Apply gains and overlay
        combined = vocal_audio.overlay(bgm_audio + bgm_gain, position=0)
        # We can also apply vocal_gain if needed
        if vocal_gain != 0.0:
            combined = combined + vocal_gain
            
        combined.export(final_output_path, format="wav")
    except Exception as e:
        print(f"Error during mixing: {e}")
        return

    print("Pipeline completed successfully! Prototype saved to data/output/final_prototype.wav")

if __name__ == "__main__":
    main()
