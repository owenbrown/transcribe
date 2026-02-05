#!/usr/bin/env python3
"""Real-time microphone transcription using Mistral's Voxtral API."""

import asyncio
import os
import signal
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.models import (
    AudioFormat,
    RealtimeTranscriptionError,
    TranscriptionStreamTextDelta,
)

SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 100
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
ENV_FILE = Path(__file__).parent.parent / ".env"


async def test_api_key(api_key: str) -> bool:
    """Test API key by connecting to the realtime transcription endpoint."""
    client = Mistral(api_key=api_key)
    audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=SAMPLE_RATE)

    async def empty_audio() -> AsyncIterator[bytes]:
        yield b"\x00" * 3200  # Single silent frame
        return

    try:
        async for event in client.audio.realtime.transcribe_stream(
            audio_stream=empty_audio(),
            model="voxtral-mini-transcribe-realtime-2602",
            audio_format=audio_format,
        ):
            pass
        return True
    except Exception:
        return False


def prompt_for_api_key() -> str:
    """Prompt user for their Mistral API key."""
    print("Get your API key at: https://console.mistral.ai/")
    return input("Enter your Mistral API key: ").strip()


async def iter_microphone(sample_rate: int) -> AsyncIterator[bytes]:
    """Yield PCM chunks from microphone using sounddevice."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        pcm_bytes = (indata * 32767).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(queue.put_nowait, pcm_bytes)

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
        blocksize=CHUNK_SAMPLES,
        callback=audio_callback,
    ):
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk


async def main() -> None:
    load_dotenv()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        api_key = prompt_for_api_key()
        if not api_key:
            print("Error: No API key provided", file=sys.stderr)
            sys.exit(1)

        print("Testing API key...", file=sys.stderr)
        if not await test_api_key(api_key):
            print("Error: Invalid API key", file=sys.stderr)
            sys.exit(1)

        ENV_FILE.write_text(f"MISTRAL_API_KEY={api_key}\n")
        print(f"Looks good. I stored it in .env\n", file=sys.stderr)

    client = Mistral(api_key=api_key)
    audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=SAMPLE_RATE)

    print("Listening... (Ctrl+C to stop)", file=sys.stderr)

    try:
        async for event in client.audio.realtime.transcribe_stream(
            audio_stream=iter_microphone(SAMPLE_RATE),
            model="voxtral-mini-transcribe-realtime-2602",
            audio_format=audio_format,
        ):
            if isinstance(event, TranscriptionStreamTextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, RealtimeTranscriptionError):
                print(f"\nError: {event}", file=sys.stderr)
    except asyncio.CancelledError:
        pass


def signal_handler(sig, frame) -> None:
    print("\nStopping...", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    asyncio.run(main())
