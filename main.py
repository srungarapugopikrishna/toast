import mimetypes
import os
import sys

from config import load_config
from utils.audio import convert_video_to_audio, convert_to_wav
from utils.silence import detect_silences
from utils.transcript import transcribe_audio, save_outputs
from utils.video_edit import build_segments, export_edited_video, get_video_length


def is_video(path):
    mime, _ = mimetypes.guess_type(path)
    return mime is not None and mime.startswith("video")


def is_audio(path):
    mime, _ = mimetypes.guess_type(path)
    return mime is not None and mime.startswith("audio")


def process_file(input_path, cfg):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    model_name = cfg["model_size"]  # base / small / medium / large

    # Create folder: output/<video>/<model>/
    out_dir = os.path.join("output", base_name, model_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 Output Folder: {out_dir}")

    # ---------- Extract / Prepare Audio ----------
    print(f"🎧 Preparing audio for: {input_path}")
    if is_video(input_path):
        wav = convert_video_to_audio(input_path)
    elif is_audio(input_path):
        wav = convert_to_wav(input_path)
    else:
        print(f"⛔ Unsupported file format: {input_path}")
        return

    # ---------- Transcription ----------
    words, sentences = transcribe_audio(wav, cfg["model_size"], cfg["language"])

    # ---------- Silence Detection ----------
    silence_cfg = cfg.get("silence", {})
    silence_threshold = silence_cfg.get("threshold", 0.02)
    silence_min_dur = silence_cfg.get("min_duration", 0.30)

    detect_silences(
        wav,
        out_dir,
        energy_threshold=silence_threshold,
        min_silence_sec=silence_min_dur
    )

    # ---------- Save Outputs ----------
    save_outputs(words, sentences, wav, out_dir)

    # ---------- Jump-cut Video Editing ----------
    sil_csv = os.path.join(out_dir, "silences.csv")
    jump_cfg = cfg.get("jumpcut", {})
    enabled = jump_cfg.get("enabled", False)

    if enabled and os.path.exists(sil_csv) and is_video(input_path):
        print("🎬 Jump-cut mode enabled → Editing video...")
        duration = get_video_length(input_path)

        min_silence = jump_cfg.get("remove_if_silence_longer_than", 1.0)
        pad_before = jump_cfg.get("pad_before", 0.10)
        pad_after = jump_cfg.get("pad_after", 0.25)

        segments = build_segments(sil_csv, duration, cfg)
        edited = export_edited_video(input_path, segments, wav, out_dir)
        print("🎞 Jump-cut video saved:", edited)
    else:
        print("⚠️ Jump-cut disabled or silences.csv missing → video not edited")

    print("✅ Finished:", input_path)


def main():
    cfg = load_config()

    if len(sys.argv) < 2:
        print("Usage: python main.py input_folder_or_file")
        return

    target = sys.argv[1]

    # Process Folder
    if os.path.isdir(target):
        for f in os.listdir(target):
            path = os.path.join(target, f)
            if is_video(path) or is_audio(path):
                process_file(path, cfg)
        return

    # Single File
    process_file(target, cfg)


if __name__ == "__main__":
    main()
