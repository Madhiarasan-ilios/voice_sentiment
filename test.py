import asyncio
import base64
from app import TranscribeAudioHandler, MyTranscriptHandler
import wave

class TestClientHandler:
    def __init__(self):
        self.transcriptions = []

    def send_transcription(self, transcription):
        self.transcriptions.append(transcription)

class TestTranscribeAudioHandler(TranscribeAudioHandler):
    def __init__(self, client_handler, region="ap-south-1"):
        super().__init__(client_handler, region)
        self._transcription_task = None

    async def start_transcription(self):
        self._stream = await self._transcribe_client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=16000,
            media_encoding="pcm"
        )
        handler = MyTranscriptHandler(self._client_handler)
        self._transcription_task = asyncio.create_task(handler.handle_events(self._stream))

    async def end_stream(self):
        if self._is_streaming:
            await self._stream.input_stream.end_stream()
            self._is_streaming = False
            if self._transcription_task:
                await self._transcription_task

def read_audio_file(file_path, chunk_size=1600):
    with wave.open(file_path, 'rb') as wf:
        if wf.getframerate() != 16000:
            raise ValueError("Audio file must be 16000 Hz")
        if wf.getsampwidth() != 2:
            raise ValueError("Audio file must be 16-bit PCM")
        while True:
            frames = wf.readframes(chunk_size)
            if not frames:
                break
            yield frames

async def test_with_audio_file(file_path):
    test_client = TestClientHandler()
    handler = TestTranscribeAudioHandler(test_client, region="ap-south-1")

    for chunk in read_audio_file(file_path):
        base64_chunk = base64.b64encode(chunk).decode('utf-8')
        await handler.receive_audio(base64_chunk, "test_id")

    await handler.end_stream()

    print("Collected transcriptions:")
    for trans in test_client.transcriptions:
        print(trans)

if __name__ == "__main__":
    asyncio.run(test_with_audio_file('FBAI_Sample_Tamil_CC_Healthcare.wav'))