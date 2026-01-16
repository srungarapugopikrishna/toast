import ffmpeg


def convert_video_to_audio(input_file, output='temp_audio.wav'):
    print("🎞 Extracting audio from video...")
    (
        ffmpeg
        .input(input_file)
        .output(output, ac=1, ar=16000, map='0:a:0')
        .overwrite_output()
        .run(quiet=True)
    )
    return output


def convert_to_wav(input_file, output="temp_audio.wav"):
    print("🎧 Converting audio -> WAV...")
    (
        ffmpeg
        .input(input_file)
        .output(output, ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )
    return output
