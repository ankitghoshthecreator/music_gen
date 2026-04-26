class HarmonyEngine:
    def __init__(self, target_bpm):
        self.target_bpm = target_bpm
        self.beats_per_bar = 4
        self.bar_duration = (60.0 / target_bpm) * self.beats_per_bar
        
    def generate_chords(self, melody_notes, total_duration):
        """
        Heuristic-based harmonizer.
        Divides the time into bars and assigns a chord based on the melody notes present.
        """
        print("Generating chords using heuristics...")
        num_bars = int((total_duration // self.bar_duration) + 1)
        chords = []
        
        for bar_idx in range(num_bars):
            start_time = bar_idx * self.bar_duration
            end_time = (bar_idx + 1) * self.bar_duration
            
            # Find notes in this bar
            bar_notes = [n['note'] for n in melody_notes if n['start'] >= start_time and n['start'] < end_time]
            
            if not bar_notes:
                # Default to C minor if empty (or hold previous chord)
                root_note = 60 if bar_idx == 0 else chords[-1]['root']
                chord_type = 'minor'
            else:
                # Simplistic heuristic: majority note as root, alternate major/minor or just use minor
                root_note = max(set(bar_notes), key=bar_notes.count)
                chord_type = 'minor' if bar_idx % 2 == 0 else 'major'
                
            chords.append({
                'root': root_note,
                'type': chord_type,
                'start': start_time,
                'end': end_time
            })
            
        return chords
