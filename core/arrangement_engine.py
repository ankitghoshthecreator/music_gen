import pretty_midi

class ArrangementEngine:
    def __init__(self, target_bpm, config=None):
        self.target_bpm = target_bpm
        self.config = config or {}
        
    def create_midi(self, melody_notes, chords):
        print("Arranging MIDI tracks...")
        midi_data = pretty_midi.PrettyMIDI(initial_tempo=self.target_bpm)
        
        # 1. Melody Track
        melody_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        melody_track = pretty_midi.Instrument(program=melody_program, name="Melody")
        
        for n in melody_notes:
            note = pretty_midi.Note(
                velocity=n.get('velocity', 100),
                pitch=n['note'],
                start=n['start'],
                end=n['end']
            )
            melody_track.notes.append(note)
        midi_data.instruments.append(melody_track)
        
        # 2. Chords / Pad Track
        pad_program = pretty_midi.instrument_name_to_program('Pad 1 (new age)')
        pad_track = pretty_midi.Instrument(program=pad_program, name="Pad Chords")
        
        for chord in chords:
            root = chord['root']
            # Simple triads
            if chord['type'] == 'major':
                intervals = [0, 4, 7]
            else:
                intervals = [0, 3, 7]
                
            for interval in intervals:
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=root + interval,
                    start=chord['start'],
                    end=chord['end']
                )
                pad_track.notes.append(note)
        midi_data.instruments.append(pad_track)
        
        # 3. Bass Track
        bass_program = pretty_midi.instrument_name_to_program('Electric Bass (finger)')
        bass_track = pretty_midi.Instrument(program=bass_program, name="Bass")
        
        for chord in chords:
            root = chord['root']
            # Bass plays root note an octave down
            bass_note = root - 12
            while bass_note < 36: bass_note += 12 # Keep in reasonable bass range
            
            note = pretty_midi.Note(
                velocity=100,
                pitch=bass_note,
                start=chord['start'],
                end=chord['end']
            )
            bass_track.notes.append(note)
        midi_data.instruments.append(bass_track)

        # 4. Violin Track (Sustained Pads)
        violin_program = pretty_midi.instrument_name_to_program('Violin')
        violin_track = pretty_midi.Instrument(program=violin_program, name="Violin Pads")
        sustain_factor = self.config.get("arrangement", {}).get("sustain_factor", 1.2)
        
        for chord in chords:
            root = chord['root']
            # Violin plays third and fifth an octave up for texture
            intervals = [4, 7] if chord['type'] == 'major' else [3, 7]
            for interval in intervals:
                note = pretty_midi.Note(
                    velocity=70,
                    pitch=root + interval + 12,
                    start=chord['start'],
                    end=chord['start'] + (chord['end'] - chord['start']) * sustain_factor
                )
                violin_track.notes.append(note)
        midi_data.instruments.append(violin_track)

        # 5. Harmonium/Organ (Arpeggios)
        organ_program = pretty_midi.instrument_name_to_program('Reed Organ')
        organ_track = pretty_midi.Instrument(program=organ_program, name="Harmonium Arps")
        
        beat_duration = 60.0 / self.target_bpm
        for chord in chords:
            root = chord['root']
            intervals = [0, 4, 7, 12] if chord['type'] == 'major' else [0, 3, 7, 12]
            
            # Simple arpeggio pattern over the chord duration
            duration = chord['end'] - chord['start']
            num_arps = int(duration / (beat_duration / 2)) # 8th notes
            
            for i in range(num_arps):
                arp_time = chord['start'] + i * (beat_duration / 2)
                pitch = root + intervals[i % len(intervals)]
                note = pretty_midi.Note(
                    velocity=60,
                    pitch=pitch,
                    start=arp_time,
                    end=arp_time + (beat_duration / 2) * 0.9
                )
                organ_track.notes.append(note)
        midi_data.instruments.append(organ_track)
        
        # 6. Drums (Simple 4-to-the-floor beat)
        drum_track = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
        beats_per_bar = 4
        bar_duration = (60.0 / self.target_bpm) * beats_per_bar
        beat_duration = 60.0 / self.target_bpm
        
        if chords:
            total_duration = chords[-1]['end']
            num_beats = int(total_duration / beat_duration)
            
            for b in range(num_beats):
                beat_time = b * beat_duration
                # Kick on every beat
                kick = pretty_midi.Note(velocity=100, pitch=36, start=beat_time, end=beat_time + 0.1)
                drum_track.notes.append(kick)
                
                # Snare on 2 and 4
                if b % 2 != 0:
                    snare = pretty_midi.Note(velocity=100, pitch=38, start=beat_time, end=beat_time + 0.1)
                    drum_track.notes.append(snare)
                    
                # Hihat every half beat
                hihat1 = pretty_midi.Note(velocity=80, pitch=42, start=beat_time, end=beat_time + 0.1)
                hihat2 = pretty_midi.Note(velocity=80, pitch=42, start=beat_time + beat_duration/2, end=beat_time + beat_duration/2 + 0.1)
                drum_track.notes.append(hihat1)
                drum_track.notes.append(hihat2)
                
        midi_data.instruments.append(drum_track)
        return midi_data
