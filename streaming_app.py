import queue
import threading
import sounddevice as sd
from google.cloud import speech
from google.oauth2 import service_account
import streamlit as st

# Assuming your Google Cloud credentials are set up in Streamlit's secrets
#credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = speech.SpeechClient()
def audio_stream_generator(q):
    """Generator function that yields audio chunks from a queue."""
    while True:
        chunk = q.get()
        if chunk is None:  # Use None as a signal to end the stream.
            break
        yield speech.StreamingRecognizeRequest(audio_content=chunk)

def stream_audio(transcript_queue, stop_event, device_index=None):
    audio_q = queue.Queue(maxsize=10)

    def audio_callback(indata, frames, time, status):
        if status:
            print("Audio Input Error:", status, file=sys.stderr)
        audio_q.put(bytes(indata))

    with sd.RawInputStream(callback=audio_callback, dtype='int16', channels=1, samplerate=16000, device=device_index):
        requests = audio_stream_generator(audio_q)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code='en-US',
        )
        streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

        try:
            responses = client.streaming_recognize(config=streaming_config, requests=requests)
            for response in responses:
                if response.error.code:
                    print("Google Speech API Error:", response.error.message)
                    break

                for result in response.results:
                    if result.is_final:
                        transcript = result.alternatives[0].transcript
                        transcript_queue.put(transcript)
                        print("Transcript updated:", transcript)  # Debugging print

                if stop_event.is_set():
                    print("Stop event triggered.")
                    break

        except Exception as e:
            print("Exception during streaming:", e)
        finally:
            audio_q.put(None)  # Signal the generator to terminate

def generate_transcript(transcript_queue):
    while True:
        transcript = transcript_queue.get()
        if transcript is None:
            break
        yield transcript

def main():
    st.title("Real-time Speech Recognition")

    transcript_queue = queue.Queue()
    stop_event = threading.Event()

    if st.button("Start Recording"):
        if 'transcribe_thread' in st.session_state and st.session_state.transcribe_thread.is_alive():
            st.warning("Recording is already in progress")
        else:
            stop_event.clear()
            st.session_state.transcribe_thread = threading.Thread(target=stream_audio, args=(transcript_queue, stop_event), daemon=True)
            st.session_state.transcribe_thread.start()
            st.success("Recording started")

    if st.button("Stop Recording"):
        stop_event.set()
        if 'transcribe_thread' in st.session_state:
            st.session_state.transcribe_thread.join()
            st.success("Recording stopped")
            transcript_queue.put(None)  # Signal the generator to terminate

    transcript = generate_transcript(transcript_queue)
    st.write_stream(transcript)

if __name__ == "__main__":
    main()