import json
import os
import re
import pandas as pd
from faster_whisper import WhisperModel

from utils.text_normalizer import normalize_sentence_text

# ------------------------------------------------------------
# Model paths
# ------------------------------------------------------------
MODEL_PATHS = {
    "base": "models/base",
    "small": "models/small",
    "medium": "models/medium",
    "large": "models/large-v3"
}

# ------------------------------------------------------------
# Whisper model loader
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Transcription
# ------------------------------------------------------------
def transcribe_audio(wav_path, model_size, language):
    model = load_whisper_model(model_size)
    lang_arg = None if language.lower() == "auto" else language

    print("🧠 Running transcription...")
    segments, info = model.transcribe(
        wav_path,
        language=lang_arg,
        vad_filter=False,        # DO NOT cut words
        word_timestamps=True
    )

    print("📝 Detected Language:", info.language)
    print("⏱ Audio Duration:", info.duration, "sec")

    words = []
    sentences = []

    for seg in segments:
        raw_text = seg.text.strip()
        sentences.append((raw_text, seg.start, seg.end))

        if seg.words:
            for w in seg.words:
                words.append((w.word, w.start, w.end, w.probability))
        else:
            # fallback approximation
            tokens = raw_text.split()
            dur = (seg.end - seg.start) / max(len(tokens), 1)
            t = seg.start
            for tok in tokens:
                words.append((tok, t, t + dur, 1.0))
                t += dur

    return words, sentences

# ------------------------------------------------------------
# Output writers (RAW + NORMALIZED)
# ------------------------------------------------------------
def save_outputs(words, sentences, wav_path, out_dir, cfg=None):
    os.makedirs(out_dir, exist_ok=True)

    # ===========================
    # RAW TRANSCRIPT (GROUND TRUTH)
    # ===========================
    pd.DataFrame(
        words,
        columns=["word", "start", "end", "confidence"]
    ).to_csv(
        os.path.join(out_dir, "transcript_raw_words.csv"),
        index=False
    )

    pd.DataFrame(
        sentences,
        columns=["sentence", "start", "end"]
    ).to_csv(
        os.path.join(out_dir, "transcript_raw_sentences.csv"),
        index=False
    )

    with open(os.path.join(out_dir, "transcript_raw.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "audio_file": wav_path,
                "words": words,
                "sentences": sentences,
            },
            f,
            indent=2,
        )

    # ===========================
    # NORMALIZED TRANSCRIPT (TEXT ONLY)
    # ===========================
    normalized_sentences = []

    if cfg:
        cleanup_cfg = cfg.get("text_cleanup", {})
        filler_words = cfg.get("fillers", {}).get("words", [])
        mode = cleanup_cfg.get("mode", "balanced")
        min_words = cleanup_cfg.get("min_meaningful_words", 3)
    else:
        filler_words = []
        mode = "balanced"
        min_words = 3

    for text, start, end in sentences:
        clean = normalize_sentence_text(
            text=text,
            filler_words=filler_words,
            min_meaningful_words=min_words,
            mode=mode,
        )
        if clean:
            normalized_sentences.append((clean, start, end))

    pd.DataFrame(
        normalized_sentences,
        columns=["sentence", "start", "end"]
    ).to_csv(
        os.path.join(out_dir, "transcript_normalized_sentences.csv"),
        index=False
    )

    with open(os.path.join(out_dir, "transcript_normalized.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "audio_file": wav_path,
                "sentences": normalized_sentences,
            },
            f,
            indent=2,
        )

# ------------------------------------------------------------
# Filler detection (AUDIO ONLY, RAW WORDS)
# ------------------------------------------------------------

NON_LEXICAL_REGEX = re.compile(
    r"^(uh+h*|um+m*|er+r*|erm+m*|ah+h*|eh+h*|hm+m*|mm+m*|mhm+|uh-?huh|yy+y*|ee+)$"
)

def detect_fillers(words, cfg):
    """
    Detect fillers from RAW word timestamps.
    Returns:
      - filler_segments: [[start, end], ...]
      - filler_debug_rows: [word, start, end, type]
    """

    fillers_cfg = cfg.get("fillers", {})
    if not fillers_cfg.get("enabled", False):
        return [], []

    lexical_fillers = {
        f.lower().strip()
        for f in fillers_cfg.get("words", [])
    }

    filler_segments = []
    filler_debug = []

    PAD = 0.02  # safety padding (seconds)

    for word, start, end, prob in words:
        # 🔧 CRITICAL FIX: remove punctuation for detection
        clean = re.sub(r"[^\w\-]", "", word).lower()

        is_lexical = clean in lexical_fillers
        is_non_lexical = bool(NON_LEXICAL_REGEX.match(clean))

        if is_lexical or is_non_lexical:
            kind = "lexical" if is_lexical else "non_lexical"

            s = max(0.0, float(start) - PAD)
            e = float(end) + PAD

            filler_segments.append([s, e])
            filler_debug.append([clean, s, e, kind])

    print(f"🧹 Detected {len(filler_segments)} fillers")
    return filler_segments, filler_debug
