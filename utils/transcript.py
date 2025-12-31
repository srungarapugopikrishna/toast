import os
import json
import pandas as pd
from faster_whisper import WhisperModel

MODEL_PATHS = {
    "base": "models/base",
    "small": "models/small",
    "medium": "models/medium",
    "large": "models/large-v3"
}

def load_whisper_model(size: str):
    model_dir = os.path.abspath(MODEL_PATHS[size])
    print(f"🔍 Loading Whisper model from: {model_dir}")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"❌ Whisper model folder missing: {model_dir}")

    return WhisperModel(
        model_dir,
        local_files_only=True,
        device="cpu",
        compute_type="int8"
    )


def transcribe_audio(wav_path, model_size, language):
    model = load_whisper_model(model_size)

    lang_arg = None if language.lower() == "auto" else language

    print("🧠 Running transcription...")
    segments, info = model.transcribe(
        wav_path,
        language=lang_arg,
        vad_filter=False,        # Avoid cutting words
        word_timestamps=True
    )

    print("📝 Detected Language:", info.language)
    print("⏱ Audio Duration:", info.duration, "sec")

    words = []
    sentences = []

    for seg in segments:
        sentences.append((seg.text, seg.start, seg.end))

        if seg.words:
            for w in seg.words:
                words.append((w.word, w.start, w.end, w.probability))
        else:
            # fallback: sentence split
            t = seg.text.split()
            ts = (seg.end - seg.start) / max(len(t), 1)
            st = seg.start
            for token in t:
                words.append((token, st, st + ts, 1.0))
                st += ts

    return words, sentences


def save_outputs(words, sentences, wav_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Create word-level CSV
    pd.DataFrame(words, columns=["word","start","end","confidence"]) \
        .to_csv(os.path.join(out_dir, "transcript_words.csv"), index=False)

    # Sentence-level CSV
    pd.DataFrame(sentences, columns=["sentence","start","end"]) \
        .to_csv(os.path.join(out_dir, "transcript_sentences.csv"), index=False)

    # JSON output
    data = {
        "audio_file": wav_path,
        "words": words,
        "sentences": sentences
    }
    json.dump(data, open(os.path.join(out_dir, "transcript.json"), "w"), indent=2)
