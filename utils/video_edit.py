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

    if not sentences:
        return [[0.0, duration]]

    jc = cfg["jumpcut"]

    pad_after = jc.get("pad_after", 0.25)
    keep_gap = jc.get("keep_gap_sec", 0.25)

    merge_gap = cfg["video"].get("merge_gap_sec", 0.40)

    # Sentence start protection window (seconds)
    SENTENCE_GUARD = cfg.get("editing", {}).get("sentence_guard_sec", 0.15)

    silences = read_silence_csv(silence_csv)

    # --------------------------------------------------
    # Sentence anchors
    # --------------------------------------------------
    sentence_ranges = [(s, e) for _, s, e in sentences]

    def overlaps_sentence_head(a, b):
        for s, e in sentence_ranges:
            head_end = min(e, s + SENTENCE_GUARD)
            if a < head_end and b > s:
                return True
        return False

    # --------------------------------------------------
    # Initial speech segments (sentence-based)
    # --------------------------------------------------
    speech = []
    for _, start, end in sentences:
        s = max(0.0, start)
        e = min(duration, end + pad_after)
        speech.append([s, e])

    # Merge overlapping speech
    merged = []
    for seg in sorted(speech):
        if not merged or seg[0] > merged[-1][1]:
            merged.append(seg)
        else:
            merged[-1][1] = max(merged[-1][1], seg[1])

    # --------------------------------------------------
    # Apply cuts (silence shrink + filler hard delete)
    # --------------------------------------------------
    final = []

    for seg_start, seg_end in merged:
        cursor = seg_start
        emitted_anything = False

        cut_zones = []

        # Silences (shrinkable)
        for s in silences:
            if s[0] > seg_start and s[1] < seg_end:
                if not overlaps_sentence_head(s[0], s[1]):
                    cut_zones.append(("silence", s))

        # Fillers (hard delete)
        for f in filler_segments:
            if f[0] > seg_start and f[1] < seg_end:
                if not overlaps_sentence_head(f[0], f[1]):
                    cut_zones.append(("filler", f))

        cut_zones.sort(key=lambda x: x[1][0])

        for kind, (cs, ce) in cut_zones:
            if cursor < cs:
                final.append([cursor, cs])
                emitted_anything = True

            if kind == "silence":
                # Shrink silence
                cursor = max(cursor, ce - keep_gap)
            else:
                # HARD DELETE filler
                cursor = ce

        # Tail
        if cursor < seg_end:
            final.append([cursor, seg_end])
            emitted_anything = True

        # Safety: never drop a full sentence
        if not emitted_anything:
            final.append([seg_start, seg_end])

    # --------------------------------------------------
    # Merge close segments
    # --------------------------------------------------
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
