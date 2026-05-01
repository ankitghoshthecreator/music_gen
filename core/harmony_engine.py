class HarmonyEngine:
    def __init__(self, target_bpm, config=None):
        self.target_bpm = target_bpm
        self.config = config or {}
        self.beats_per_bar = 4
        self.chord_change_beats = self.config.get("arrangement", {}).get("chord_change_interval_beats", 4)
        self.bar_duration = (60.0 / target_bpm) * self.chord_change_beats
        
    def generate_chords(self, melody_notes, total_duration, detected_key=None):
        """
        Heuristic-based harmonizer that uses detected key for better consonance.
        """
        if detected_key and len(detected_key) == 3:
            key_root, key_mode, key_conf = detected_key
        else:
            key_root, key_mode = detected_key if detected_key else (0, 'major')
        root_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key_root]
        print(f"Generating chords for Key: {root_name} {key_mode}...")

        num_bars = int((total_duration // self.bar_duration) + 1)
        chords = []
        
        # Diatonic roots for major and minor scales (0-11)
        if key_mode == 'major':
            # I, ii, iii, IV, V, vi, vii°
            diatonic_roots = [(key_root + offset) % 12 for offset in [0, 2, 4, 5, 7, 9, 11]]
            chord_types = ['major', 'minor', 'minor', 'major', 'major', 'minor', 'diminished']
        else:
            # i, ii°, III, iv, v, VI, VII
            diatonic_roots = [(key_root + offset) % 12 for offset in [0, 2, 3, 5, 7, 8, 10]]
            chord_types = ['minor', 'diminished', 'major', 'minor', 'minor', 'major', 'major']

        for bar_idx in range(num_bars):
            start_time = bar_idx * self.bar_duration
            end_time = (bar_idx + 1) * self.bar_duration
            
            bar_notes = [n['note'] for n in melody_notes if n['start'] >= start_time and n['start'] < end_time]
            
            if not bar_notes:
                # Default to I or i
                root_note = 60 + key_root
                chord_type = key_mode
            else:
                # Find the most frequent pitch class in the bar
                pitch_classes = [p % 12 for p in bar_notes]
                most_freq_pc = max(set(pitch_classes), key=pitch_classes.count)
                
                # Pick the diatonic chord that has this PC as its root, or closest
                if most_freq_pc in diatonic_roots:
                    idx = diatonic_roots.index(most_freq_pc)
                    root_note = 60 + most_freq_pc
                    chord_type = chord_types[idx]
                else:
                    # Fallback to I/i if not diatonic
                    root_note = 60 + key_root
                    chord_type = key_mode
                
            chords.append({
                'root': root_note,
                'type': chord_type,
                'start': start_time,
                'end': end_time
            })
            
        return chords
