from pathlib import Path
import yt_dlp

output_folder = Path("videos")
output_folder.mkdir()

channels = ['https://www.youtube.com/@GujarRandhak']

ydl_opts = {
    'format': 'm4a/bestaudio/best',
    'outtmpl': str(output_folder / '%(title)s.%(ext)s'),
    'postprocessors': [{  # Extract audio using ffmpeg
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }]
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    error_code = ydl.download(channels)

