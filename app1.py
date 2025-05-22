import streamlit as st
import boto3
import json
import os
import time
import html
import re
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
import string

# AWS Clients
transcribe = boto3.client('transcribe')
translate = boto3.client('translate')
s3 = boto3.client('s3')

# Bedrock LLM
llm = ChatBedrock(model_id="meta.llama3-70b-instruct-v1:0")

# Constants
BASE_DIR = "./audio_processing"
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEMP_BUCKET = "audio-testfiles"
MAX_CHARS = 4000

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sentiment analysis prompt
SENTIMENT_SYSTEM_PROMPT = """
You are a sentiment-analysis assistant. Your SOLE purpose is to output valid JSON that strictly conforms to the following schema. DO NOT include any conversational text, explanations, or any other content outside of the JSON object.

{
  "text": string,
  "overall_sentiment": string,  // "positive", "negative", "neutral", "mixed"
  "sentiment_score": number,  // confidence score (0–1) for the overall_sentiment
  "positive_words": [string],
  "negative_words": [string],
  "neutral_words": [string],
  "mixed_words": [string],
  "key_influencers": [string],
  "aspect_sentiment": {
    "<aspect_name>": number  // sentiment score for each aspect, from -1 (very negative) to 1 (very positive)
  },
  "reasoning": string
}
"""

# Function to strip punctuation and convert to lowercase
def strip_punctuation(word):
    return word.translate(str.maketrans('', '', string.punctuation)).lower()

# Function to build highlighted HTML for a single line
def build_highlighted_line(line_parts):
    html_parts = []
    for part in line_parts:
        if isinstance(part, str):
            html_parts.append(f'<span class="speaker">{html.escape(part)}</span> ')
        else:
            sentiment = part['sentiment'].lower()
            html_parts.append(f'<span class="{sentiment}">{html.escape(part["text"])}</span>')
    return ' '.join(html_parts)

# Function to build highlighted HTML with speaker labels
def build_highlighted_html_with_speakers(processed_segments):
    highlighted_lines = []
    for speaker_turn in processed_segments:
        if not speaker_turn:
            continue
        speaker_label = speaker_turn[0]['speaker']
        line_parts = [speaker_label] + [item for item in speaker_turn[1:]]
        highlighted_content = build_highlighted_line(line_parts)
        full_line_html = f'<div class="line">{highlighted_content}</div>'
        highlighted_lines.append(full_line_html)
    return '\n'.join(highlighted_lines)

# Function to translate text while preserving speaker labels
def translate_with_speaker_labels(text, source_lang, target_lang='en'):
    if source_lang == target_lang:
        return text, source_lang
    lines = text.split('\n')
    translated_lines = []
    for line in lines:
        if line.startswith('Speaker'):
            colon_pos = line.find(':')
            if colon_pos != -1:
                speaker = line[:colon_pos+1]
                spoken_text = line[colon_pos+1:].strip()
                translation = translate.translate_text(Text=spoken_text, SourceLanguageCode=source_lang, TargetLanguageCode=target_lang)
                translated_spoken_text = translation['TranslatedText']
                translated_line = f"{speaker} {translated_spoken_text}"
            else:
                translated_line = line
        else:
            translation = translate.translate_text(Text=line, SourceLanguageCode=source_lang, TargetLanguageCode=target_lang)
            translated_line = translation['TranslatedText']
        translated_lines.append(translated_line)
    return '\n'.join(translated_lines), source_lang

# Function to detect language and translate if necessary
def detect_and_translate(text):
    try:
        lang_prompt = f"Detect the dominant language of this text and return the language code (e.g., 'en' for English, 'ta' for Tamil) in JSON: ```{text[:5000]}```"
        messages = [
            SystemMessage(content="You are a language detection assistant. Output only valid JSON with the language code."),
            HumanMessage(content=lang_prompt)
        ]
        lang_response = llm.invoke(messages)
        lang_data = json.loads(lang_response.content)
        dominant_language = lang_data.get('language_code', 'en')
    except:
        dominant_language = 'en'
    if dominant_language != 'en':
        processed_text, _ = translate_with_speaker_labels(text, dominant_language)
    else:
        processed_text = text
    return processed_text, dominant_language

# Function to split text into chunks
def split_into_chunks(text, max_chars=MAX_CHARS):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    return splitter.split_text(text)

# Function to process audio and perform sentiment analysis
def process_audio(audio_file_path, filename):
    safe_filename = re.sub(r'[^0-9a-zA-Z._-]', '-', filename)
    job_name = f"transcribe-job-{safe_filename}-{int(time.time())}"
    media_format = 'wav' if filename.endswith('.wav') else 'mp3'

    # Upload audio to S3
    s3.upload_file(audio_file_path, TEMP_BUCKET, filename)

    # Start transcription job with speaker labels
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': f's3://{TEMP_BUCKET}/{filename}'},
        MediaFormat=media_format,
        IdentifyLanguage=True,
        Settings={
            'ShowSpeakerLabels': True,
            'MaxSpeakerLabels': 2
        },
        OutputBucketName=TEMP_BUCKET,
        OutputKey=f'transcriptions/{job_name}.json'
    )

    # Wait for transcription to complete
    with st.spinner("Transcribing audio..."):
        while True:
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            if job_status in ['COMPLETED', 'FAILED']:
                break
            time.sleep(5)

    if job_status == 'FAILED':
        raise RuntimeError("Transcription failed")

    # Download transcription
    transcription_key = f'transcriptions/{job_name}.json'
    transcription_file = os.path.join(OUTPUT_DIR, f"{job_name}.json")
    s3.download_file(TEMP_BUCKET, transcription_key, transcription_file)

    # Read transcription
    with open(transcription_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract transcript with speaker labels
    full_transcript_text = ""
    segments_with_speakers = []
    current_speaker_segment = []
    current_speaker_label = None
    for item in data['results']['items']:
        content = item['alternatives'][0]['content']
        item_type = item['type']
        if item_type == 'pronunciation':
            speaker_label = item.get('speaker_label', 'Unknown Speaker')
            if current_speaker_label is None or speaker_label != current_speaker_label:
                if current_speaker_segment:
                    segments_with_speakers.append(current_speaker_segment)
                current_speaker_segment = [{'speaker': speaker_label}]
                current_speaker_label = speaker_label
            current_speaker_segment.append({'word': content})
        elif item_type == 'punctuation' and current_speaker_segment:
            current_speaker_segment[-1]['word'] += content
    if current_speaker_segment:
        segments_with_speakers.append(current_speaker_segment)

    # Reconstruct transcript for LLM
    reconstructed_transcript_for_llm = []
    for segment in segments_with_speakers:
        raw_speaker = segment[0]['speaker']  # e.g., "spk_0"
        # Normalize speaker name to "speaker_0", "speaker_1", etc.
        speaker = re.sub(r'^spk_', 'speaker_', raw_speaker.lower())
        words_in_segment = [item['word'] for item in segment[1:]]
        reconstructed_transcript_for_llm.append(f"{speaker}: { ' '.join(words_in_segment) }")

    full_transcript_text = "\n".join(reconstructed_transcript_for_llm)

    # Detect language and translate if necessary
    processed_text, detected_language = detect_and_translate(full_transcript_text)

    # Save transcriptions
    transcript_path = os.path.join(OUTPUT_DIR, f"{job_name}_transcript.txt")
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    original_tamil_text = None
    if detected_language == 'ta':
        tamil_transcript_path = os.path.join(OUTPUT_DIR, f"{job_name}_original_tamil_transcript.txt")
        with open(tamil_transcript_path, 'w', encoding='utf-8') as f:
            f.write(full_transcript_text)
        original_tamil_text = full_transcript_text

    # Split full transcript into chunks
    full_chunks = split_into_chunks(processed_text)

    # Initialize
    chunk_results = []
    sentiment_scores = []
    master_positive = set()
    master_negative = set()
    master_neutral = set()
    master_mixed = set()

    # Process each full chunk
    for chunk in full_chunks:
        messages = [
            SystemMessage(content=SENTIMENT_SYSTEM_PROMPT),
            HumanMessage(content=f"""Analyze this text for sentiment. Determine the overall sentiment as 'positive', 'negative', 'neutral', or 'mixed', and provide a confidence score (0–1). Identify which words contribute to each sentiment type (positive, negative, neutral, mixed), highlight the key influencer words, identify aspect-level sentiment scores (e.g., billing, service) from -1 to 1, and briefly explain why the overall sentiment is what it is. Output **only** the JSON object.
Text: "{chunk}" """)
        ]
        response = llm.invoke(messages)
        try:
            sentiment_data = json.loads(response.content)
            if all(key in sentiment_data for key in ["text", "overall_sentiment", "sentiment_score", "positive_words", "negative_words", "neutral_words", "mixed_words", "key_influencers", "aspect_sentiment", "reasoning"]):
                chunk_results.append(sentiment_data)
                # Calculate score
                sentiment = sentiment_data['overall_sentiment'].lower()
                confidence = sentiment_data['sentiment_score']
                score = confidence if sentiment == 'positive' else -confidence if sentiment == 'negative' else 0
                sentiment_scores.append(score)
                # Add to master sets
                master_positive.update([strip_punctuation(word) for word in sentiment_data['positive_words']])
                master_negative.update([strip_punctuation(word) for word in sentiment_data['negative_words']])
                master_neutral.update([strip_punctuation(word) for word in sentiment_data['neutral_words']])
                master_mixed.update([strip_punctuation(word) for word in sentiment_data['mixed_words']])
            else:
                st.warning(f"Invalid sentiment response for chunk starting with '{chunk[:50]}'")
        except Exception as e:
            st.warning(f"Error processing chunk: {e}")

    # Calculate final score
    if sentiment_scores:
        avg_score = sum(sentiment_scores) / len(sentiment_scores)
        final_score = 4.5 * avg_score + 5.5
    else:
        final_score = 5.5

    # Parse processed_text into translated_segments
    translated_segments = []
    lines = processed_text.split('\n')
    for line in lines:
        if ':' in line:
            parts = line.split(': ', 1)
            if len(parts) == 2:
                speaker = parts[0]
                text = parts[1]
                words = text.split()  # split by whitespace
                translated_segments.append([{'speaker': speaker}] + [{'text': word} for word in words])

    # Build processed_segments_with_sentiment using master lists
    processed_segments_with_sentiment = []
    for segment in translated_segments:
        segment_with_sentiment = [part.copy() for part in segment]  # deep copy
        for i in range(1, len(segment_with_sentiment)):  # start from 1, since 0 is speaker
            text_dict = segment_with_sentiment[i]
            text = text_dict['text']
            text_clean = strip_punctuation(text)
            if text_clean in master_positive:
                text_dict['sentiment'] = 'positive'
            elif text_clean in master_negative:
                text_dict['sentiment'] = 'negative'
            elif text_clean in master_mixed:
                text_dict['sentiment'] = 'mixed'
            elif text_clean in master_neutral:
                text_dict['sentiment'] = 'neutral'
            else:
                text_dict['sentiment'] = 'neutral'
        processed_segments_with_sentiment.append(segment_with_sentiment)

    # Build highlighted_html
    highlighted_html = build_highlighted_html_with_speakers(processed_segments_with_sentiment)

    # Cleanup S3
    s3.delete_object(Bucket=TEMP_BUCKET, Key=filename)
    s3.delete_object(Bucket=TEMP_BUCKET, Key=transcription_key)

    return {
        "final_score": round(final_score, 2),
        "chunks": chunk_results,
        "detected_language": detected_language,
        "highlighted_html": highlighted_html,
        "original_tamil_text": original_tamil_text,
        "transcript_path": transcript_path
    }

# Streamlit App
st.title("Audio Sentiment Analysis App")
st.write("Upload an audio file (.wav or .mp3) to analyze its sentiment.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    filename = uploaded_file.name
    if filename.endswith('.wav'):
        audio_format = 'audio/wav'
    elif filename.endswith('.mp3'):
        audio_format = 'audio/mpeg'
    else:
        st.error("Please upload a .wav or .mp3 file.")
        st.stop()
    st.audio(uploaded_file, format=audio_format)
    if st.button("Start Processing"):
        try:
            with st.spinner("Processing audio... This may take a few minutes."):
                file_path = os.path.join(INPUT_DIR, filename)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                result_data = process_audio(file_path, filename)
            st.success(f"✅ Final Sentiment Score: {result_data['final_score']} (1 = very negative, 10 = very positive)")
            st.write(f"📄 Transcription saved to `{result_data['transcript_path']}`")
            st.write(f"Detected Language: {result_data['detected_language']}")
            if result_data['detected_language'] != 'en':
                st.write(f"The text has been translated from {result_data['detected_language']} to English.")
                if result_data['detected_language'] == 'ta':
                    tamil_transcript_path = os.path.join(OUTPUT_DIR, f"{filename.replace('.', '-')}_original_tamil_transcript.txt")
                    st.write(f"Original Tamil transcription saved to `{tamil_transcript_path}`")
                    st.write("### Original Tamil Transcription")
                    st.write(result_data['original_tamil_text'])
            st.subheader("Sentiment Analysis Results")
            for i, chunk in enumerate(result_data['chunks'], 1):
                st.write(f"**Conversation Chunk {i}:**")
                st.write(f"Text: {chunk['text'][:100]}..." if len(chunk['text']) > 100 else f"Text: {chunk['text']}")
                st.write(f"Overall Sentiment: {chunk['overall_sentiment']}")
                st.write(f"Confidence Score: {chunk['sentiment_score']}")
                st.write(f"Key Influencers: {', '.join(chunk['key_influencers'])}")
                if chunk['aspect_sentiment']:
                    st.write("Aspect Sentiments:")
                    for aspect, score in chunk['aspect_sentiment'].items():
                        st.write(f"  - {aspect}: {score}")
                st.write(f"Reasoning: {chunk['reasoning']}")
                st.write("---")
            st.markdown("""
            <style>
            .line { margin-bottom: 10px; }
            .speaker { font-weight: bold; }
            .positive { color: green; }
            .negative { color: red; }
            .neutral { color: gray; }
            .mixed { color: orange; }
            </style>
            """, unsafe_allow_html=True)
            st.markdown("### Highlighted Transcription")
            st.markdown(result_data['highlighted_html'], unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")