
import re
from pathlib import Path

import whisper

from utils.audio_processor import download_audio, chunk_audio


ROOT_PATH = Path(__file__).resolve().parent.parent
TRANSCRIPTS_FOLDER = ROOT_PATH / "transcripts"
TRANSCRIPTS_FOLDER.mkdir(parents=True, exist_ok=True)


# Function 1: Load Whisper model
def load_whisper_model(model_name="small"):

    print("Loading Whisper model...")

    model = whisper.load_model(model_name)

    print("Whisper model loaded successfully!")

    return model


# Function 2: Transcribe audio chunks
def transcribe_audio(model, chunks):

    print("Starting transcription...")

    transcripts = []

    for i, chunk_path in enumerate(chunks):

        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        result = model.transcribe(
            chunk_path,
            task="transcribe",
            fp16=False
        )

        text = result["text"].strip()

        transcripts.append(text)

    print("Transcription completed!")

    return transcripts


# Main pipeline function
def transcribe_video(url: str, chunk_min: int = 2):

    # Download audio
    audio_path, title = download_audio(url)

    # Split audio into chunks
    chunks = chunk_audio(
        wav_path=audio_path,
        title=title,
        chunk_min=chunk_min
    )

    # Load model once
    model = load_whisper_model()

    # Transcribe all chunks
    transcripts = transcribe_audio(
        model=model,
        chunks=chunks
    )

    # Create safe filename
    safe_title = re.sub(r'[<>:"/\\|?*]', "_", title)

    output_path = TRANSCRIPTS_FOLDER / f"{safe_title}.txt"

    # Save transcript
    with open(output_path, "w", encoding="utf-8") as file:

        for i, text in enumerate(transcripts):

            file.write(f"\n--- Chunk {i + 1} ---\n")
            file.write(text + "\n")

    print("Transcript saved:", output_path)

    return transcripts, str(output_path)
