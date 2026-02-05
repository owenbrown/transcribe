# Transcribe

Real-time microphone transcription using Mistral's [Voxtral Mini](https://mistral.ai/news/voxtral-transcribe-2) API.

## Setup

1. Get a Mistral API key from [console.mistral.ai](https://console.mistral.ai/)

2. Create a `.env` file with your key:
   ```
   MISTRAL_API_KEY=your_key_here
   ```

3. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

```bash
uv run python app/transcribe.py
```

Speak into your microphone and see transcriptions appear in real-time. Press `Ctrl+C` to stop.
