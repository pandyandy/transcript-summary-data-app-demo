import streamlit as st
import sounddevice as sd
import soundfile as sf
import vertexai
import cv2
import io
import vertexai.preview.generative_models as generative_models

from google.cloud import speech, storage
from vertexai.generative_models import GenerativeModel, Part, FinishReason

# Function to capture audio
def capture_audio(duration=15, fs=16000):
    sd.default.samplerate = fs
    sd.default.channels = 1
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    #, blocking=True)
    sd.wait()
    return audio_data

# Function to capture video frame
#def capture_video_frame(camera_id=0):
#    cap = cv2.VideoCapture(camera_id)
#    ret, frame = cap.read()
#    cap.release()
#    return frame

# Function to upload to Google Cloud Storage
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

# Function to transcribe audio using Google Cloud Speech-to-Text
def transcribe_audio(audio_file_path, language_code="en-US"):
    client = speech.SpeechClient()
    
    with io.open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
    )
    response = client.recognize(config=config, audio=audio)
    transcript = ' '.join([alternative.transcript for result in response.results for alternative in result.alternatives])
    
    return transcript
    
# Function to generate summary using Gemini
def generate(content):
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
        contents=f"Provide a brief summary (1-2 sentences) of the following transcript: {content}",
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    output_text = "".join(response.text for response in responses)
    return output_text


# Streamlit app
def main():
    st.title("Audio Recording and Summarization")

    recording_state = st.session_state.get('recording', False)

    if st.button("Start Recording") or recording_state:
    #while True:
        
        st.session_state['recording'] = True
        st.write("Recording... Click 'Stop Recording' to end.")
        
        # Capture audio
        audio_data = capture_audio()
        audio_filename = "temp_audio.wav"
        sf.write(audio_filename, audio_data, 16000)

        # Upload to Google Cloud Storage
        upload_blob("vojta-transcript-demo-test", audio_filename, "audio.wav")

        # Transcribe audio
        transcript = transcribe_audio(audio_filename)
        
        # Generate summary
        summary = generate(transcript)

        # Display results
        st.write("**Transcript:**", transcript)
        st.write("**Summary:**", summary)

        # Stop recording if button is pressed
        if st.button("Stop Recording"):
            st.session_state['recording'] = False
            st.write("Recording stopped.")

if __name__ == "__main__":
    main()