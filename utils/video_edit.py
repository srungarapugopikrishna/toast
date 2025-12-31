import os

import ffmpeg
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips,
    concatenate_audioclips
)


def get_video_length(path):
    probe = ffmpeg.probe(path)
    return float(probe["format"]["duration"])


def build_segments(silence_csv, duration, cfg):
    import pandas as pd
    df = pd.read_csv(silence_csv)

    pad_before = cfg.get("pad", {}).get("before", 0.10)
    pad_after = cfg.get("pad", {}).get("after", 0.30)
    min_clip = cfg.get("video", {}).get("min_clip_sec", 0.80)
    merge_gap = cfg.get("video", {}).get("merge_gap_sec", 0.40)

    raw_segments = []
    prev_end = 0.0

    for _, row in df.iterrows():
        start = float(row["start"])
        end = float(row["end"])

        seg_start = max(0.0, prev_end - pad_before)
        seg_end = min(start + pad_after, duration)

        if seg_end - seg_start >= min_clip:
            raw_segments.append((seg_start, seg_end))

        prev_end = end

    # last chunk
    if duration - prev_end >= min_clip:
        raw_segments.append((max(prev_end - pad_before, 0), duration))

    # ---- merge adjacent chunks if too close ----
    merged = []
    for seg in raw_segments:
        if not merged:
            merged.append(seg)
            continue
        last_start, last_end = merged[-1]
        cur_start, cur_end = seg

        if cur_start - last_end <= merge_gap:
            merged[-1] = (last_start, max(last_end, cur_end))
        else:
            merged.append(seg)

    print("🧮 RAW SEGMENTS:", raw_segments)
    print("🤝 MERGED SEGMENTS:", merged)
    return merged



def export_edited_video(input_video, segments, wav_path, out_dir):
    print("🎞 Building final edited video... replacing audio track")

    video = VideoFileClip(input_video)
    audio = AudioFileClip(wav_path)

    clips = []
    audio_clips = []

    for start, end in segments:
        clips.append(video.subclip(start, end))
        audio_clips.append(audio.subclip(start, end))

    # Concatenate video + audio properly
    final_video = concatenate_videoclips(clips, method="compose")
    final_audio = concatenate_audioclips(audio_clips)

    # Attach trimmed audio to trimmed video
    final = final_video.set_audio(final_audio)

    output_file = os.path.join(out_dir, "edited_jumpcut.mp4")

    final.write_videofile(
        output_file,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        temp_audiofile=os.path.join(out_dir, "_temp_audio.m4a"),
        remove_temp=True,
        fps=video.fps  # preserve original video fps
    )

    video.close()
    audio.close()

    print("🎬 Final jumpcut exported:", output_file)
    return output_file
