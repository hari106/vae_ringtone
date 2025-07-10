import pretty_midi
import numpy as np
from constants import MAX_TIMESTEPS, TIME_STEP, PITCH_RANGE
import os

# --- Utilities ---
def midi_to_embedding(file_path, time_step=TIME_STEP, pitch_range=PITCH_RANGE):
    midi_data = pretty_midi.PrettyMIDI(file_path)
    total_time = midi_data.get_end_time()
    time_bins = np.arange(0, total_time, time_step)
    piano_roll = np.zeros((len(time_bins), pitch_range[1] - pitch_range[0]))
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                start_idx = int(note.start / time_step)
                end_idx = int(note.end / time_step)
                if pitch_range[0] <= note.pitch < pitch_range[1]:
                    pitch_idx = note.pitch - pitch_range[0]
                    piano_roll[start_idx:end_idx, pitch_idx] = note.velocity / 127.0
    return piano_roll

def process_midi_dataset(folder):
    embeddings = []
    for fname in os.listdir(folder):
        if fname.endswith(".mid") or fname.endswith(".midi"):
            path = os.path.join(folder, fname)
            emb = midi_to_embedding(path)
            embeddings.append(emb)
    return embeddings

def pad_embeddings(embeddings, max_len=MAX_TIMESTEPS):
    padded = []
    for e in embeddings:
        if e.shape[0] > max_len:
            padded.append(e[:max_len])
        else:
            pad_width = max_len - e.shape[0]
            padded_e = np.pad(e, ((0, pad_width), (0, 0)))
            padded.append(padded_e)
    return np.array(padded)

