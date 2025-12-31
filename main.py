import os
import sys
import mimetypes
from config import load_config
from utils.audio import convert_video_to_audio, convert_to_wav
from utils.transcript import transcribe_audio, save_outputs
from utils.silence import detect_silences


def is_video(path):
    mime, _ = mimetypes.guess_type(path)
    return mime is not None and mime.startswith("video")


def is_audio(path):
    mime, _ = mimetypes.guess_type(path)
    return mime is not None and mime.startswith("audio")


def process_file(input_path, cfg):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    model_name = cfg["model_size"]  # medium, large, base, etc.

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

    detect_silences(wav, out_dir,
                    energy_threshold=silence_threshold,
                    min_silence_sec=silence_min_dur)

    # ---------- Save Outputs ----------
    save_outputs(words, sentences, wav, out_dir)

    print("✅ Finished:", input_path)


def main():
    cfg = load_config()

    if len(sys.argv) < 2:
        print("Usage: python main.py input_folder_or_file")
        return

    target = sys.argv[1]

    # ----- Process Folder of Files -----
    if os.path.isdir(target):
        for f in os.listdir(target):
            path = os.path.join(target, f)
            if is_video(path) or is_audio(path):
                process_file(path, cfg)
        return

    # ----- Process Single File -----
    process_file(target, cfg)


if __name__ == "__main__":
    main()
