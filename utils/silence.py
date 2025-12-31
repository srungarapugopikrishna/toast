import numpy as np
import soundfile as sf
import pandas as pd
import os

def detect_silences(wav_path, out_dir, energy_threshold=0.02, min_silence_sec=0.30):
    print(f"🔊 Detecting silences... threshold={energy_threshold}, min_dur={min_silence_sec}")

    data, sr = sf.read(wav_path)
    if len(data.shape) > 1:
        data = data[:, 0]

    energy = np.abs(data)
    silence_mask = energy < energy_threshold

    silences = []
    start_idx = None

    for i, silent in enumerate(silence_mask):
        if silent and start_idx is None:
            start_idx = i
        elif not silent and start_idx is not None:
            duration = (i - start_idx) / sr
            if duration >= min_silence_sec:
                silences.append((start_idx / sr, i / sr))
            start_idx = None

    if start_idx is not None:
        end = len(data) / sr
        duration = end - (start_idx / sr)
        if duration >= min_silence_sec:
            silences.append((start_idx / sr, end))

    print(f"🧮 Found {len(silences)} silence regions")

    df = pd.DataFrame(silences, columns=["start", "end"])
    csv_path = os.path.join(out_dir, "silences.csv")
    df.to_csv(csv_path, index=False)

    print(f"💾 Saved silences → {csv_path}")
