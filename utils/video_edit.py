import os
import csv
import ffmpeg
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips

def get_video_length(path):
    probe = ffmpeg.probe(path)
    return float(probe["format"]["duration"])

def read_silence_csv(csv_path):
    silences = []
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        next(r)  # skip header
        for row in r:
            silences.append([float(row[0]), float(row[1])])
    return silences


def build_segments(silence_csv, duration, sentences, cfg):
    jc = cfg["jumpcut"]
    shrink_threshold = jc.get("shrink_long_silence_sec", 1.0)
    keep_gap = jc.get("keep_gap_sec", 0.25)
    pad_before = jc.get("pad_before", 0.10)
    pad_after = jc.get("pad_after", 0.25)
    min_clip = cfg["video"].get("min_clip_sec", 0.80)
    merge_gap = cfg["video"].get("merge_gap_sec", 0.40)

    # Build raw speech segments (from transcript)
    speech_segments = []
    for text, start, end in sentences:
        start = max(0, float(start) - pad_before)
        end = min(duration, float(end) + pad_after)
        speech_segments.append([start, end])

    # Merge word-based speech segments
    merged = []
    for seg in sorted(speech_segments, key=lambda x: x[0]):
        if not merged or seg[0] > merged[-1][1]:
            merged.append(seg)
        else:
            merged[-1][1] = max(merged[-1][1], seg[1])

    silences = read_silence_csv(silence_csv)
    print("🔊 Forced silences:", silences)

    # Now force split merged segments using silence.csv
    final_segments = []
    for seg in merged:
        seg_start, seg_end = seg
        inside = [s for s in silences if s[0] > seg_start and s[1] < seg_end]

        if not inside:
            final_segments.append(seg)
            continue

        cursor = seg_start
        for sil in inside:
            st, en = sil
            dur = en - st
            if dur >= shrink_threshold:
                # Add speech before silence
                if cursor < st and (st - cursor) >= min_clip:
                    final_segments.append([cursor, st])

                # shrink silence to small gap
                cursor = en - keep_gap
            # else ignore short silence fully
        # End last section
        if cursor < seg_end:
            final_segments.append([cursor, seg_end])

    # Merge if two segments too close
    compact = []
    for seg in final_segments:
        if not compact or seg[0] - compact[-1][1] > merge_gap:
            compact.append(seg)
        else:
            compact[-1][1] = seg[1]

    print("🧩 FINAL MERGED SEGMENTS:", compact)
    return compact


def export_edited_video(input_video, segments, wav_path, out_dir):
    print("🎞 Building final edited video... replacing audio track")

    video = VideoFileClip(input_video)
    audio = AudioFileClip(wav_path)

    clips = []
    audio_clips = []

    for start, end in segments:
        clips.append(video.subclip(start, end))
        audio_clips.append(audio.subclip(start, end))

    final_video = concatenate_videoclips(clips, method="compose")
    final_audio = concatenate_audioclips(audio_clips)
    final = final_video.set_audio(final_audio)

    output = os.path.join(out_dir, "edited_jumpcut.mp4")
    final.write_videofile(
        output,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        temp_audiofile=os.path.join(out_dir, "_temp_audio.m4a"),
        remove_temp=True,
        fps=video.fps
    )

    video.close()
    audio.close()
    print("🎬 Final jumpcut exported:", output)
    return output
