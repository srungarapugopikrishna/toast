import os
import csv
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


def read_silence_csv(csv_path):
    silences = []
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            silences.append([float(row[0]), float(row[1])])
    return silences


def build_segments(silence_csv, duration, sentences, filler_segments, cfg):
    jc = cfg["jumpcut"]
    pad_before = jc.get("pad_before", 0.10)
    pad_after = jc.get("pad_after", 0.25)

    shrink_silence = jc.get("remove_if_silence_longer_than", 1.0)
    keep_gap = jc.get("keep_gap_sec", 0.25)

    min_clip = cfg["video"].get("min_clip_sec", 0.80)
    merge_gap = cfg["video"].get("merge_gap_sec", 0.40)

    filler_gap = cfg["fillers"].get("shrink_to", 0.08)

    # 1️⃣ Build speech segments from transcript
    speech = []
    for _, start, end in sentences:
        s = max(0, start - pad_before)
        e = min(duration, end + pad_after)
        speech.append([s, e])

    # 2️⃣ Merge overlapping speech
    merged = []
    for seg in sorted(speech):
        if not merged or seg[0] > merged[-1][1]:
            merged.append(seg)
        else:
            merged[-1][1] = max(merged[-1][1], seg[1])

    silences = read_silence_csv(silence_csv)

    # 3️⃣ Apply silence + filler shrinking
    final = []
    for seg_start, seg_end in merged:
        cursor = seg_start

        cut_zones = [
            *[s for s in silences if s[0] > seg_start and s[1] < seg_end],
            *[f for f in filler_segments if f[0] > seg_start and f[1] < seg_end],
        ]
        cut_zones.sort()

        for st, en in cut_zones:
            dur = en - st
            shrink = keep_gap if dur >= shrink_silence else filler_gap

            if cursor < st and (st - cursor) >= min_clip:
                final.append([cursor, st])

            cursor = max(cursor, en - shrink)

        if cursor < seg_end and (seg_end - cursor) >= min_clip:
            final.append([cursor, seg_end])

    # 4️⃣ Merge close segments (prevents overlap)
    compact = []
    for seg in final:
        if not compact or seg[0] - compact[-1][1] > merge_gap:
            compact.append(seg)
        else:
            compact[-1][1] = seg[1]

    print("🧩 FINAL MERGED SEGMENTS:", compact)
    return compact


def export_edited_video(input_video, segments, wav_path, out_dir):
    video = VideoFileClip(input_video)
    audio = AudioFileClip(wav_path)

    vclips, aclips = [], []
    for s, e in segments:
        vclips.append(video.subclip(s, e))
        aclips.append(audio.subclip(s, e))

    final_v = concatenate_videoclips(vclips, method="compose")
    final_a = concatenate_audioclips(aclips)

    final = final_v.set_audio(final_a)
    out = os.path.join(out_dir, "edited_jumpcut.mp4")

    final.write_videofile(
        out,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        fps=video.fps,
        temp_audiofile=os.path.join(out_dir, "_temp.m4a"),
        remove_temp=True
    )

    video.close()
    audio.close()
    return out
