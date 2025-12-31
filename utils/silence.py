import librosa
import numpy as np
import pandas as pd
import os

def detect_silences(wav_path, out_dir,
                    energy_threshold=0.02,
                    min_silence_sec=0.30):

    wav, sr = librosa.load(wav_path, sr=None)
    frame_len = int(sr * 0.02)  # 20ms frame
    hop = frame_len

    energies = []
    for i in range(0, len(wav), hop):
        frame = wav[i:i+frame_len]
        eng = np.sqrt(np.mean(frame**2))
        energies.append(eng)

    times = np.arange(len(energies)) * (frame_len / sr)
    silences = []
    start = None

    for t, e in zip(times, energies):
        if e < energy_threshold:
            if start is None:
                start = t
        else:
            if start is not None:
                end = t
                dur = end - start
                if dur >= min_silence_sec:
                    silences.append((start, end, dur))
                start = None

    if start is not None:
        end = times[-1]
        dur = end - start
        if dur >= min_silence_sec:
            silences.append((start, end, dur))

    df = pd.DataFrame(silences, columns=["start", "end", "duration"])
    df.to_csv(os.path.join(out_dir, "silences.csv"), index=False)

    return silences
