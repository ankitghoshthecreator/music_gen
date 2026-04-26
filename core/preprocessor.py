import numpy as np
import librosa

class AudioPreprocessor:
    def __init__(self, config):
        self.config = config
        
    def reduce_noise(self, y, sr):
        """Simplistic noise gate based on noise_floor_db."""
        floor_db = self.config.get("noise_reduction", {}).get("noise_floor_db", -40.0)
        floor_amp = librosa.db_to_amplitude(floor_db)
        y_cleaned = np.where(np.abs(y) < floor_amp, 0.0, y)
        return y_cleaned

    def apply_vad(self, y, sr):
        """Trims absolute silence or low energy segments."""
        yt, index = librosa.effects.trim(y, top_db=60) 
        return yt
        
    def extract_melody_midi(self, y, sr, target_bpm):
        """Extracts pitch and converts it to discrete MIDI notes on a BPM grid."""
        pitch_config = self.config.get("pitch", {})
        fmin = pitch_config.get("fmin_hz", 80.0)
        fmax = pitch_config.get("fmax_hz", 800.0)
        
        # Extract pitch using librosa.pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=fmin, 
            fmax=fmax, 
            sr=sr,
            frame_length=2048,
            hop_length=512
        )
        
        # Convert f0 (Hz) to MIDI note numbers
        midi_notes_raw = librosa.hz_to_midi(f0)
        
        hop_duration = 512 / sr
        notes = []
        
        current_note = None
        start_time = 0.0
        
        # Basic heuristic to group contiguous voiced frames into discrete notes
        for i, is_voiced in enumerate(voiced_flag):
            if is_voiced and not np.isnan(midi_notes_raw[i]):
                note_num = int(round(midi_notes_raw[i]))
                if current_note is None:
                    current_note = note_num
                    start_time = i * hop_duration
                elif current_note != note_num:
                    end_time = i * hop_duration
                    if end_time - start_time > 0.1: # Min duration 100ms
                        notes.append({'note': current_note, 'start': start_time, 'end': end_time})
                    current_note = note_num
                    start_time = i * hop_duration
            else:
                if current_note is not None:
                    end_time = i * hop_duration
                    if end_time - start_time > 0.1:
                        notes.append({'note': current_note, 'start': start_time, 'end': end_time})
                    current_note = None

        return notes

    def process(self, y, sr, target_bpm):
        """Run the full preprocessing pipeline."""
        print("Cleaning audio and extracting melody...")
        y_clean = self.reduce_noise(y, sr)
        y_vad = self.apply_vad(y_clean, sr)
        melody_notes = self.extract_melody_midi(y_vad, sr, target_bpm)
        return y_vad, melody_notes
