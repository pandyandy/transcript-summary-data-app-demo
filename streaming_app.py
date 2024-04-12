import queue
import threading
import sounddevice as sd
from google.cloud import speech
from google.oauth2 import service_account
import streamlit as st
import sys
import time
import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.generative_models import GenerativeModel, Part, FinishReason

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


def generate_summary(content):
    vertexai.init(project="keboola-ai", location="us-central1")
    model = GenerativeModel("gemini-1.5-pro-preview-0409")
    #model = GenerativeModel("gemini-1.5-pro-latest")

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }
    
    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    
    responses = model.generate_content(
        contents=f"""
        Create a concise (1-2 sentences) summary from given transcript. 
        Write the summary in the third person, do NOT mention the transcript in the summary.  
        Include a dash at the beginning so that it can be used as a bullet point.

        Transcript: 
        {content}
        """,
        generation_config=generation_config,
        #safety_settings=safety_settings,
        stream=True,
    )

    output_text = "".join(response.text for response in responses)
    return output_text

def generate_transcript(transcript_queue):
    while True:
        transcript = transcript_queue.get()
        if transcript is None:
            break
        yield transcript

def get_summary(summary_queue):
    while True:
        summary = summary_queue.get()
        if summary is None:
            break
        yield summary

def summary_update(transcript_queue, summary_queue, stop_event):
    accumulated_text = ""
    while not stop_event.is_set():
        try:
                transcript = transcript_queue.get_nowait()
                if transcript:
                    print(f"Received transcript: {transcript}")  # Debugging print
                accumulated_text += " " + transcript
        except queue.Empty:
            if accumulated_text:
                print(f"Accumulated transcript for summary: {accumulated_text}")  # Debugging print
                summary = generate_summary(accumulated_text)
                if summary:
                    print(f"Generated summary: {summary}")  # Debugging print
                summary_queue.put(summary)
                accumulated_text = ""
            time.sleep(30)  

def main():
    st.title("Real-time Speech Recognition")

    transcript_queue = queue.Queue()
    summary_queue = queue.Queue()
    stop_event = threading.Event()

    if st.button("Start Recording"):
        if 'transcribe_thread' in st.session_state and st.session_state.transcribe_thread.is_alive():
            st.warning("Recording is already in progress")
        else:
            stop_event.clear()
            st.session_state.transcribe_thread = threading.Thread(target=stream_audio, args=(transcript_queue, stop_event), daemon=True)
            st.session_state.summary_thread = threading.Thread(target=summary_update, args=(transcript_queue, summary_queue, stop_event), daemon=True)
            st.session_state.transcribe_thread.start()
            st.session_state.summary_thread.start()
            st.success("Recording started")

    if st.button("Stop Recording"):
        stop_event.set()
        if 'transcribe_thread' in st.session_state:
            st.session_state.transcribe_thread.join()
            st.session_state.summary_thread.join()
            st.success("Recording stopped")
            transcript_queue.put(None)  # Signal the generator to terminate
    
    st.subheader("Summary 🤪")
    summary = get_summary(summary_queue)
    st.write_stream(summary)

if __name__ == "__main__":
    main()