from pathlib import Path
import argparse
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment


ROOT = Path(__file__).resolve().parent


def transcribe(client, path, prompt=None):
    options = {
        "model": "whisper-1",
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"],
    }
    if prompt:
        options["prompt"] = prompt

    with path.open("rb") as audio_file:
        return client.audio.transcriptions.create(
            file=audio_file, **options
        ).model_dump()


def split_audio(audio, folder, seconds=60):
    if seconds <= 0:
        raise ValueError("Chunk duration must be positive.")

    folder.mkdir(parents=True, exist_ok=True)
    chunks = []

    for index, start in enumerate(range(0, len(audio), seconds * 1000)):
        chunk = audio[start:start + seconds * 1000]
        chunk = chunk.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        path = folder / f"chunk_{index:03d}.wav"
        chunk.export(path, format="wav")

        if path.stat().st_size >= 25_000_000:
            raise ValueError("Chunk too large; use a shorter duration.")

        chunks.append((path, start / 1000))

    return chunks


def timestamp(seconds, separator="."):
    hours, remainder = divmod(round(seconds * 1000), 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return (
        f"{hours:02}:{minutes:02}:{secs:02}"
        f"{separator}{milliseconds:03}"
    )


def save_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_transcription(segments, folder):
    text = " ".join(s["text"] for s in segments)
    save_json(
        folder / "transcription.json",
        {"text": text, "segments": segments},
    )

    lines = [
        f'[{timestamp(s["start"])} - {timestamp(s["end"])}] {s["text"]}'
        for s in segments
    ]
    (folder / "transcription.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    blocks = [
        f'{i}\n'
        f'{timestamp(s["start"], ",")} --> {timestamp(s["end"], ",")}\n'
        f'{s["text"]}\n'
        for i, s in enumerate(segments, 1)
    ]
    (folder / "transcription.srt").write_text(
        "\n".join(blocks), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio", type=Path,
        default=ROOT / "audio" / "project_nimbus_meeting.mp3",
    )
    parser.add_argument("--chunk-seconds", type=int, default=60)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY in .env.")

    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)

    client = OpenAI(api_key=key)
    audio = AudioSegment.from_file(args.audio)
    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    output = ROOT / "outputs"
    chunk_folder = ROOT / "chunks"
    output.mkdir(exist_ok=True)
    chunk_folder.mkdir(exist_ok=True)

    sample_path = chunk_folder / "sample_30_seconds.wav"
    sample = audio[:30_000]
    sample = sample.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    sample.export(sample_path, format="wav")

    prompt = (
        "Project Nimbus, Kubernetes migration, PostgreSQL latency, "
        "Q3 OKRs, Priya, Marcus, authentication."
    )
    basic = transcribe(client, sample_path)
    guided = transcribe(client, sample_path, prompt)

    for name, result in [
        ("basic_transcription.txt", basic),
        ("prompted_transcription.txt", guided),
    ]:
        (output / name).write_text(result["text"], encoding="utf-8")

    save_json(output / "comparison.json", {
        "audio_file": args.audio.name,
        "sample_duration_seconds": len(sample) / 1000,
        "prompt": prompt,
        "without_prompt": basic["text"],
        "with_prompt": guided["text"],
        "identical": basic["text"] == guided["text"],
    })

    chunks = split_audio(audio, chunk_folder, args.chunk_seconds)
    segments = []

    for index, (path, offset) in enumerate(chunks, 1):
        print(f"Transcribing chunk {index}/{len(chunks)}...")
        result = transcribe(client, path)
        for segment in result.get("segments", []):
            segments.append({
                "start": segment["start"] + offset,
                "end": segment["end"] + offset,
                "text": segment["text"].strip(),
            })

    if not segments:
        raise ValueError("No speech segments were returned.")

    export_transcription(segments, output)
    print("Outputs saved in:", output)


if __name__ == "__main__":
    main()