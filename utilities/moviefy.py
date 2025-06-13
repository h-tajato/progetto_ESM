from moviepy.editor import ImageClip, AudioFileClip

def moviefy(im_path:str, audio_path:str, out_path:str):
    audio = AudioFileClip(audio_path)

    clip = ImageClip(im_path, duration=audio.duration)
    clip = clip.set_audio(audio)

    clip.write_videofile(out_path, fps=24, codec="libx264")