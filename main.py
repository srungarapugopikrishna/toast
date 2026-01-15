import mimetypes
import os
import sys
from config import load_config
from utils.audio import convert_video_to_audio, convert_to_wav
from utils.silence import detect_silences
from utils.transcript import transcribe_audio, save_outputs
from utils.video_edit import build_segments, export_edited_video, get_video_length
from utils.transcript import detect_fillers
from utils.repetition import detect_repetition_segments


from utils.transcript import detect_fillers



def is_video(path):
    mime, _ = mimetypes.guess_type(path)
    return mime and mime.startswith("video")

def is_audio(path):
    mime, _ = mimetypes.guess_type(path)
    return mime and mime.startswith("audio")


def process_file(input_path, cfg):
    base = os.path.splitext(os.path.basename(input_path))[0]
    model = cfg["model_size"]
    out_dir = os.path.join("output", base, model)
    os.makedirs(out_dir, exist_ok=True)

    print(f"📁 Output Folder: {out_dir}")

    if is_video(input_path):
        wav = convert_video_to_audio(input_path)
    elif is_audio(input_path):
        wav = convert_to_wav(input_path)
    else:
        print("❌ Unsupported file")
        return

    words, sentences = transcribe_audio(wav, model, cfg["language"])

    silence_cfg = cfg.get("silence", {})
    detect_silences(
        wav,
        out_dir,
        energy_threshold=silence_cfg.get("threshold", 0.02),
        min_silence_sec=silence_cfg.get("min_duration", 0.30)
    )

    save_outputs(words, sentences, wav, out_dir)

    sil_csv = os.path.join(out_dir, "silences.csv")
    jump_cfg = cfg.get("jumpcut", {})
    if jump_cfg.get("enabled", False) and os.path.exists(sil_csv) and is_video(input_path):
        duration = get_video_length(input_path)
        # filler_segments = detect_fillers(words, cfg)
        #
        # segments = build_segments(
        #     sil_csv,
        #     duration,
        #     sentences,
        #     filler_segments,
        #     cfg
        # )

        # Existing filler detection
        filler_segments = detect_fillers(words, cfg)

        # 🔁 NEW: continuous repetition removal
        repetition_segments = detect_repetition_segments(
            words,
            max_ngram=cfg.get("repetition", {}).get("max_ngram", 4)
        )

        # Merge both as silence-like cuts
        all_cut_segments = filler_segments + repetition_segments

        segments = build_segments(
            sil_csv,
            duration,
            sentences,
            all_cut_segments,
            cfg
        )

        export_edited_video(input_path, segments, wav, out_dir)
    else:
        print("⚠️ Jumpcut disabled — original video unedited")

    print("✅ Finished:", input_path)


def main():
    cfg = load_config()
    if len(sys.argv) < 2:
        print("Usage: python main.py file")
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
