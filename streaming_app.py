import queue
import streamlit as st
import sounddevice as sd
import numpy as np
import threading
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account

# Initialize Google Cloud Speech Client
# Note: Ensure your GCP credentials are correctly configured for this to work.
#credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = speech.SpeechClient()

def audio_stream_generator(q):
    """A generator function that yields audio chunks from the queue."""
    while True:
        chunk = q.get()
        if chunk is None:  # Use None as a sentinel value to stop the generator.
            return
        yield chunk

def stream_audio(transcript_queue):
    audio_q = queue.Queue()

    def callback(indata, frames, time, status):
        """This is called for each audio chunk from the microphone."""
        audio_q.put(np.ndarray.tobytes(indata))

    with sd.InputStream(callback=callback, dtype='int16', channels=1, samplerate=16000):
        print("Recording started...")

        stream = audio_stream_generator(audio_q)

        requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in stream)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )
        streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

        responses = client.streaming_recognize(config=streaming_config, requests=requests)

        for response in responses:
            if not response.results:
                continue
            result = response.results[0]
            if not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript
            transcript_queue.put(transcript)

        audio_q.put(None)  # Signal to generator to stop

# Streamlit app
def main():
    st.title("Real-time Audio Transcription")

    transcript_queue = queue.Queue()

    if 'transcription_thread' not in st.session_state:
        st.session_state.transcription_thread = None

    start_button, stop_button = st.columns(2)
    with start_button:
        if st.button("Start Recording"):
            if st.session_state.transcription_thread is None or not st.session_state.transcription_thread.is_alive():
                st.session_state.transcription_thread = threading.Thread(target=stream_audio, args=(transcript_queue,), daemon=True)
                st.session_state.transcription_thread.start()

    with stop_button:
        if st.button("Stop Recording"):
            if st.session_state.transcription_thread is not None:
                transcript_queue.put(None)  # Stop the audio stream generator
                st.session_state.transcription_thread.join()
                st.session_state.transcription_thread = None

    if not transcript_queue.empty():
        transcript = transcript_queue.get()
        st.write(transcript)

if __name__ == "__main__":
    main()