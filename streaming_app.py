import streamlit as st
import sounddevice as sd
from google.cloud import speech
from google.oauth2 import service_account
import numpy as np
import queue

# Streamlit app setup
def main():
    st.title("Real-Time Speech Transcription")

    if 'transcript' not in st.session_state:
        st.session_state['transcript'] = ""

    start_button, stop_button = st.columns(2)
    with start_button:
        if st.button("Start Recording"):
            start_transcription()

    with stop_button:
        if st.button("Stop Recording"):
            stop_transcription()

    st.text_area("Transcription", value=st.session_state['transcript'], height=300)

def audio_stream_generator(buffer):
    """Generator function that continuously yields audio chunks."""
    while True:
        chunk = buffer.get()
        if chunk is None:
            break
        yield chunk

def transcribe_streaming():
    """Streams transcription of the real-time audio."""
    buffer = queue.Queue()
    def callback(indata, frames, time, status):
        buffer.put(bytes(indata))

    # Audio stream configuration
    stream = sd.RawInputStream(callback=callback, dtype='int16', channels=1, samplerate=16000)
    stream.start()

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_stream_generator(buffer))
    responses = client.streaming_recognize(config=streaming_config, requests=requests)

    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        st.session_state['transcript'] += result.alternatives[0].transcript + "\n"
        st.experimental_rerun()

    buffer.put(None)
    stream.stop()
    stream.close()

def start_transcription():
    """Start the transcription process."""
    st.session_state['transcript'] = ""
    st.experimental_rerun()
    transcribe_streaming()

def stop_transcription():
    """Placeholder for stopping the transcription, actual stop mechanism needs integration."""
    pass

if __name__ == "__main__":
    main()
