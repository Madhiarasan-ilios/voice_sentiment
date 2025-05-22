import asyncio
import base64
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler

class MyTranscriptHandler(TranscriptResultStreamHandler):
    def __init__(self, client_handler):
        self._client_handler = client_handler

    async def handle_transcript_event(self, transcript_event):
        if transcript_event.event_type == "TranscriptEvent":
            # Get the latest transcription result
            latest_result = transcript_event.transcript.results[-1]
            latest_alternative = latest_result.alternatives[0]
            transcript = latest_alternative.transcript
            # Send the transcription back to the client
            await self._client_handler.send_transcription(transcript)

class TranscribeAudioHandler:
    def __init__(self, client_handler, region="ap-south-1"):
        self._client_handler = client_handler
        self._transcribe_client = TranscribeStreamingClient(region=region)
        self._is_streaming = False
        self._stream = None

    async def start_transcription(self):
        self._stream = await self._transcribe_client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=16000,
            media_encoding="pcm"
        )
        # Start handling transcription events
        handler = MyTranscriptHandler(self._client_handler)
        asyncio.create_task(handler.handle_events(self._stream))

    async def receive_audio(self, content, audio_id):
        if not self._is_streaming:
            self._is_streaming = True
            await self.start_transcription()
        # Decode base64 audio data
        audio_data = base64.b64decode(content)
        # Send audio chunk to Transcribe stream
        await self._stream.input_stream.send_audio_event(audio_chunk=audio_data)

    async def end_stream(self):
        if self._is_streaming:
            await self._stream.input_stream.end_stream()
            self._is_streaming = False