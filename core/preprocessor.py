import numpy as np
import librosa
import tempfile
import os
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

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
        """Extracts pitch and converts it to discrete MIDI notes."""
        pitch_config = self.config.get("pitch", {})
        method = pitch_config.get("method", "pyin")

        if method == "basic-pitch":
            return self._extract_with_basic_pitch(y, sr)
        else:
            return self._extract_with_pyin(y, sr)

    def _extract_with_basic_pitch(self, y, sr):
        """Extracts melody using Spotify's basic-pitch model."""
        import soundfile as sf
        bp_config = self.config.get("basic_pitch", {})
        
        # basic-pitch inference.predict expects a file path
        # We save to a temporary wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            tmp_path = tmp_audio.name
            tmp_audio.close() # Close so soundfile can write to it
            sf.write(tmp_path, y, sr)
            
        try:
            model_output, midi_data, note_events = predict(
                tmp_path,
                onset_threshold=bp_config.get("onset_threshold", 0.5),
                frame_threshold=bp_config.get("frame_threshold", 0.3),
                minimum_note_length=bp_config.get("minimum_note_length_ms", 100),
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        notes = []
        amp_threshold = bp_config.get("amplitude_threshold", 0.1) # Confidence/Noise filter
        
        for start, end, pitch, amp, hz in note_events:
            if amp < amp_threshold:
                continue
                
            # Map amplitude (0-1) to MIDI velocity (0-127)
            velocity = int(min(127, max(0, amp * 127)))
            
            notes.append({
                'note': int(round(pitch)),
                'start': start,
                'end': end,
                'velocity': velocity,
                'amplitude': float(amp)
            })
        
        notes.sort(key=lambda x: x['start'])
        return notes

    def detect_key(self, notes):
        """
        Simple key detection based on pitch frequency distribution.
        Returns the root note (0-11), mode (major/minor), and confidence.
        """
        pitch_config = self.config.get("pitch", {})
        conf_threshold = pitch_config.get("key_confidence_threshold", 0.4)
        fallback_root = pitch_config.get("fallback_key_root", 0)
        fallback_mode = pitch_config.get("fallback_key_mode", "major")

        if not notes:
            return fallback_root, fallback_mode, 0.0
            
        chroma_counts = np.zeros(12)
        for n in notes:
            pitch = n['note']
            duration = n['end'] - n['start']
            chroma_counts[pitch % 12] += duration
            
        # Normalize chroma_counts
        if np.sum(chroma_counts) > 0:
            chroma_counts = chroma_counts / np.sum(chroma_counts)
            
        # Krumhansl-Schmuckler profiles (normalized)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        major_profile /= np.sum(major_profile)
        minor_profile /= np.sum(minor_profile)
        
        best_corr = -1.0
        best_key = (fallback_root, fallback_mode)
        
        for i in range(12):
            sh_major = np.roll(major_profile, i)
            sh_minor = np.roll(minor_profile, i)
            
            corr_major = np.corrcoef(chroma_counts, sh_major)[0, 1]
            corr_minor = np.corrcoef(chroma_counts, sh_minor)[0, 1]
            
            if corr_major > best_corr:
                best_corr = corr_major
                best_key = (i, 'major')
            if corr_minor > best_corr:
                best_corr = corr_minor
                best_key = (i, 'minor')
                
        if best_corr < conf_threshold:
            print(f"Low key detection confidence ({best_corr:.2f} < {conf_threshold}). Using fallback.")
            return fallback_root, fallback_mode, best_corr
            
        return best_key[0], best_key[1], best_corr

    def _extract_with_pyin(self, y, sr):
        """Original librosa.pyin based extraction (legacy)."""
        pitch_config = self.config.get("pitch", {})
        fmin = pitch_config.get("fmin_hz", 80.0)
        fmax = pitch_config.get("fmax_hz", 800.0)
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=fmin, 
            fmax=fmax, 
            sr=sr,
            frame_length=2048,
            hop_length=512
        )
        
        midi_notes_raw = librosa.hz_to_midi(f0)
        hop_duration = 512 / sr
        notes = []
        current_note = None
        start_time = 0.0
        
        for i, is_voiced in enumerate(voiced_flag):
            if is_voiced and not np.isnan(midi_notes_raw[i]):
                note_num = int(round(midi_notes_raw[i]))
                if current_note is None:
                    current_note = note_num
                    start_time = i * hop_duration
                elif current_note != note_num:
                    end_time = i * hop_duration
                    if end_time - start_time > 0.1:
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
