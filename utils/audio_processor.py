import os
from pathlib import Path
import yt_dlp
from pydub import AudioSegment

ROOT_PATH = Path(__file__).resolve().parent.parent
DOWNLOAD_FOLDER = ROOT_PATH / "downloads"
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


def download_audio(url: str):
    print("Downloading audio...")

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(DOWNLOAD_FOLDER / "%(title)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "postprocessor_args": {
            "ffmpegextractaudio": ["-ar", "16000", "-ac", "1"]
        },
        "noplaylist": True,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        wav_path = Path(ydl.prepare_filename(info)).with_suffix(".wav")

    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    print("Audio downloaded:", wav_path)
    print("Video title:", info["title"])

    return str(wav_path), info["title"]


def chunk_audio(wav_path: str, title: str, chunk_min=10):
    print("Splitting audio into chunks...")
    print("Chunk duration:", chunk_min, "minute(s)")

    audio = AudioSegment.from_wav(wav_path)
    chunk_size = chunk_min * 60 * 1000
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_size)):
        chunk = audio[start:start + chunk_size]
        safe_title = Path(wav_path).stem
        chunk_path = DOWNLOAD_FOLDER / f"{safe_title}_chunk_{i}.wav"

        chunk.export(str(chunk_path), format="wav")
        chunks.append(str(chunk_path))

        print("Created:", chunk_path)
        print("File size:", chunk_path.stat().st_size, "bytes")

    print("Total chunks created:", len(chunks))
    return chunks


if __name__ == "__main__":
    wav_path, title = download_audio(
        url="https://www.youtube.com/watch?v=VSFuqMh4hus"
    )

    chunks = chunk_audio(
        wav_path=wav_path,
        title=title,
        chunk_min=1
    )

    print("\nAudio path:", wav_path)
    print("Chunk files:", chunks)