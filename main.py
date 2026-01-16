import mimetypes
import os
import sys

from config import load_config
from utils.audio import convert_video_to_audio, convert_to_wav
from utils.filler_debug import write_fillers_csv
from utils.silence import detect_silences
from utils.transcript import (
    transcribe_audio,
    save_outputs,
    detect_fillers,
)
from utils.video_edit import (
    build_segments,
    export_edited_video,
    get_video_length,
)


# ------------------------------------------------------------
# File type helpers
# ------------------------------------------------------------
def is_video(path):
    mime, _ = mimetypes.guess_type(path)
    return mime and mime.startswith("video")


def is_audio(path):
    mime, _ = mimetypes.guess_type(path)
    return mime and mime.startswith("audio")


# ------------------------------------------------------------
# Core processing
# ------------------------------------------------------------
def process_file(input_path, cfg):
    base = os.path.splitext(os.path.basename(input_path))[0]
    model = cfg["model_size"]

    out_dir = os.path.join("output", base, model)
    os.makedirs(out_dir, exist_ok=True)

    print(f"📁 Output Folder: {out_dir}")

    # -------------------------------
    # Audio extraction
    # -------------------------------
    if is_video(input_path):
        wav = convert_video_to_audio(input_path)
    elif is_audio(input_path):
        wav = convert_to_wav(input_path)
    else:
        print("❌ Unsupported file")
        return

    # -------------------------------
    # Transcription (RAW)
    # -------------------------------
    words, sentences = transcribe_audio(
        wav,
        model,
        cfg["language"]
    )

    # -------------------------------
    # Silence detection
    # -------------------------------
    silence_cfg = cfg.get("silence", {})
    detect_silences(
        wav,
        out_dir,
        energy_threshold=silence_cfg.get("threshold", 0.02),
        min_silence_sec=silence_cfg.get("min_duration", 0.30),
    )

    # -------------------------------
    # Save transcripts (RAW + NORMALIZED)
    # -------------------------------
    save_outputs(words, sentences, wav, out_dir, cfg)

    # -------------------------------
    # Jumpcut + filler removal
    # -------------------------------
    sil_csv = os.path.join(out_dir, "silences.csv")
    jump_cfg = cfg.get("jumpcut", {})

    if (
        jump_cfg.get("enabled", False)
        and os.path.exists(sil_csv)
        and is_video(input_path)
    ):
        duration = get_video_length(input_path)

        # Detect fillers from RAW words
        filler_segments, filler_debug = detect_fillers(words, cfg)

        # Always write fillers.csv (even if empty)
        write_fillers_csv(out_dir, filler_debug)
        print(f"🧹 Fillers detected: {len(filler_segments)}")

        # Build final cut timeline
        segments = build_segments(
            sil_csv,
            duration,
            sentences,          # RAW sentences ONLY
            filler_segments,
            cfg,
        )

        export_edited_video(
            input_path,
            segments,
            wav,
            out_dir,
        )
    else:
        print("⚠️ Jumpcut disabled — original video unedited")

    print("✅ Finished:", input_path)


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
def main():
    cfg = load_config()

    if len(sys.argv) < 2:
        print("Usage: python main.py <file_or_folder>")
        return

    target = sys.argv[1]

    if os.path.isdir(target):
        for f in os.listdir(target):
            path = os.path.join(target, f)
            if is_video(path) or is_audio(path):
                process_file(path, cfg)
    else:
        process_file(target, cfg)


if __name__ == "__main__":
    main()
