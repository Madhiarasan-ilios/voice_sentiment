import time
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import pyaudio
import websockets.sync.client
import boto3
from botocore.awsrequest import AWSRequest
from botocore.auth import SigV4Auth
import wave
import argparse

# --- Configuration ---
REGION = 'ap-south-1'
LANGUAGE_CODE = 'en-US'
SAMPLE_RATE = 16000
CHUNK_SIZE_MS = 100
FORMAT = pyaudio.paInt16
CHANNELS = 1
BITS_PER_SAMPLE = 16

# Calculate chunk size in frames and bytes
FRAMES_PER_CHUNK = int(SAMPLE_RATE * (CHUNK_SIZE_MS / 1000))
BYTES_PER_CHUNK = FRAMES_PER_CHUNK * (BITS_PER_SAMPLE // 8)

# --- Global variables ---
audio_queue = queue.Queue()
stop_event = threading.Event()
transcription_callback = None
executor = ThreadPoolExecutor(max_workers=3)
mic_future = None
sender_future = None
receiver_future = None
websocket_client = None

# --- AWS Authentication ---
def create_presigned_url():
    service_name = 'transcribe'
    http_method = 'GET'
    endpoint = f'wss://transcribestreaming.{REGION}.amazonaws.com:8443'
    uri = f'/stream-transcription-websocket?language-code={LANGUAGE_CODE}&media-encoding=pcm&sample-rate={SAMPLE_RATE}'
    session = boto3.Session()
    credentials = session.get_credentials()
    if not credentials:
        raise Exception("AWS credentials not found. Please configure them.")
    request = AWSRequest(method=http_method, url=endpoint + uri)
    sigv4 = SigV4Auth(credentials, service_name, session.region_name)
    sigv4.add_auth(request)
    return request.url

# --- Microphone Audio Capture ---
def microphone_audio_capture():
    print(f"Starting microphone capture thread. Sample rate: {SAMPLE_RATE} Hz, chunk size: {CHUNK_SIZE_MS} ms ({BYTES_PER_CHUNK} bytes)")
    p_audio = pyaudio.PyAudio()
    audio_stream = p_audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=FRAMES_PER_CHUNK)
    try:
        while not stop_event.is_set():
            audio_data = audio_stream.read(FRAMES_PER_CHUNK, exception_on_overflow=False)
            audio_queue.put(audio_data)
            time.sleep(0.001)
    finally:
        audio_stream.stop_stream()
        audio_stream.close()
        p_audio.terminate()
        print("Microphone audio capture thread stopped.")

# --- File Audio Capture ---
def file_audio_capture(audio_file):
    print(f"Reading from WAV file: {audio_file}")
    with wave.open(audio_file, "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("WAV file must be mono")
        if wav_file.getsampwidth() != 2:  # 16 bits = 2 bytes
            raise ValueError("WAV file must be 16-bit")
        if wav_file.getframerate() != SAMPLE_RATE:
            raise ValueError(f"WAV file must have sample rate of {SAMPLE_RATE} Hz")
        while not stop_event.is_set():
            audio_data = wav_file.readframes(FRAMES_PER_CHUNK)
            if not audio_data:
                break
            audio_queue.put(audio_data)
            time.sleep(CHUNK_SIZE_MS / 1000)
    # After reading the file, set stop_event to stop other threads
    stop_event.set()
    print("File reading complete. Stopping recording.")

# --- WebSocket Sender ---
def websocket_sender_thread():
    global websocket_client
    print("Starting WebSocket sender thread.")
    if not websocket_client:
        return
    try:
        while not stop_event.is_set():
            try:
                audio_chunk = audio_queue.get(timeout=0.1)
                message = {'AudioEvent': {'AudioChunk': audio_chunk}}
                websocket_client.send(json.dumps(message).encode('utf-8'))
            except queue.Empty:
                pass
    finally:
        print("WebSocket sender thread stopped.")

# --- WebSocket Receiver ---
def websocket_receiver_thread():
    global websocket_client
    print("Starting WebSocket receiver thread.")
    if not websocket_client:
        return
    try:
        while not stop_event.is_set():
            try:
                message = websocket_client.recv(timeout=0.1)
                response = json.loads(message)
                if 'Transcript' in response and 'Results' in response['Transcript']:
                    for result in response['Transcript']['Results']:
                        alt = result.get('Alternatives', [None])[0]
                        if alt:
                            transcript_content = alt.get('Transcript')
                            if transcript_content and transcription_callback:
                                transcription_callback(transcript_content)
            except websockets.exceptions.WebSocketTimeoutException:
                pass
            except websockets.exceptions.ConnectionClosedOK:
                stop_event.set()
                break
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"Receiver: WebSocket connection closed unexpectedly: {e}")
                stop_event.set()
                break
            except Exception as e:
                print(f"Error receiving transcription result: {e}")
                stop_event.set()
                break
    finally:
        print("WebSocket receiver thread stopped.")

# --- Control Functions ---
def start_recording(callback, audio_source='microphone', audio_file=None):
    global transcription_callback, mic_future, sender_future, receiver_future, websocket_client
    if is_recording_active():
        stop_recording()
    transcription_callback = callback
    stop_event.clear()
    presigned_url = create_presigned_url()
    print(f"Connecting to Transcribe WebSocket: {presigned_url}")
    try:
        websocket_client = websockets.sync.client.connect(presigned_url)
        if audio_source == 'microphone':
            mic_future = executor.submit(microphone_audio_capture)
        elif audio_source == 'file' and audio_file:
            mic_future = executor.submit(file_audio_capture, audio_file)
        else:
            raise ValueError("Invalid audio source or missing audio file")
        sender_future = executor.submit(websocket_sender_thread)
        receiver_future = executor.submit(websocket_receiver_thread)
        print("Recording started.")
    except Exception as e:
        print(f"Failed to start recording: {e}")
        stop_recording()

def stop_recording():
    global mic_future, sender_future, receiver_future, websocket_client
    if not is_recording_active():
        return
    print("Stopping recording...")
    stop_event.set()
    if mic_future:
        mic_future.result(timeout=5)
    if sender_future:
        sender_future.result(timeout=5)
    if receiver_future:
        receiver_future.result(timeout=5)
    if websocket_client:
        websocket_client.close()
        websocket_client = None
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break
    print("Recording stopped.")

def is_recording_active():
    if mic_future and not mic_future.done():
        return True
    if sender_future and not sender_future.done():
        return True
    if receiver_future and not receiver_future.done():
        return True
    return False

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time speech transcription using AWS Transcribe')
    parser.add_argument('--audio_source', choices=['microphone', 'file'], default='microphone', help='Audio source: microphone or file')
    parser.add_argument('--audio_file', type=str, help='Path to WAV file (required if audio_source is file)')
    args = parser.parse_args()

    def my_transcription_callback(transcript_text):
        print(f"Callback received: {transcript_text}")

    try:
        if args.audio_source == 'file' and not args.audio_file:
            print("Error: --audio_file is required when --audio_source is file")
            exit(1)
        start_recording(my_transcription_callback, audio_source=args.audio_source, audio_file=args.audio_file)
        while is_recording_active():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nUser interrupted.")
    finally:
        stop_recording()
        executor.shutdown(wait=True)
        print("Application exited.")