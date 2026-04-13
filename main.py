import argparse
from pathlib import Path
from utils.config_loader import load_config
from utils.audio_io import load_audio, save_audio
from core.preprocessor import AudioPreprocessor
from core.bgm_model import BGMGenerator
from core.mixer import TrackMixer

def main():
    parser = argparse.ArgumentParser(description="BGM Generation Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input vocal wav file")
    parser.add_argument("--output", type=str, default="data/output/final_mix.wav", help="Path to output mixed wav file")
    args = parser.parse_args()

    print(f"Loading configuration...")
    config = load_config()

    print(f"Loading input audio from {args.input}...")
    target_sr = config["audio"]["sample_rate"]
    y_raw, sr = load_audio(args.input, target_sr=target_sr, mono=config["audio"]["mono"])

    # 1. Preprocess Vocals
    preprocessor = AudioPreprocessor(config)
    y_vocal, vocal_features = preprocessor.process(y_raw, sr)

    # 2. Generate BGM from Features
    bgm_gen = BGMGenerator(config)
    y_bgm = bgm_gen.generate_bgm(vocal_features, target_length_samples=len(y_vocal), sr=sr)

    # 3. Mix
    mixer = TrackMixer(config)
    y_mix = mixer.mix(y_vocal, y_bgm)

    # 4. Save
    print(f"Saving final mix to {args.output}...")
    save_audio(args.output, y_mix, sr=sr)
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
