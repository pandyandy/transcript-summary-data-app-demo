import streamlit as st
import cv2
import numpy as np
import sounddevice as sd
from google.cloud import speech
import io
import os
from google.cloud import storage

# Function to capture audio
def capture_audio(duration=30, fs=16000):
    sd.default.samplerate = fs
    sd.default.channels = 1
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio_data

# Function to capture video frame
def capture_video_frame(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    ret, frame = cap.read()
    cap.release()
    return frame

# Function to upload to Google Cloud Storage
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

# Function to transcribe audio using Google Cloud Speech-to-Text
def transcribe_audio(audio_data, language_code="en-US"):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_data.tobytes())
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
    )
    response = client.recognize(config=config, audio=audio)
    return response.results[0].alternatives[0].transcript

# Function to generate summary using Gemini
def generate_summary(text):
    # Replace this with your actual API call to Gemini
    # This is a placeholder for demonstration purposes
    summary = f"**Summary:**\n* {text}"
    return summary

# Streamlit app
def main():
    st.title("Audio/Video Recording and Summarization")

    if st.button("Start Recording"):
        while True:
            # Capture audio and video
            audio_data = capture_audio()
            frame = capture_video_frame()

            # Save audio and image
            audio_filename = "temp_audio.wav"
            image_filename = "temp_image.jpg"
            sd.write(audio_filename, audio_data, 16000)
            cv2.imwrite(image_filename, frame)

            # Upload to Google Cloud Storage (replace with your bucket name)
            upload_blob("your-bucket-name", audio_filename, "audio.wav")
            upload_blob("your-bucket-name", image_filename, "image.jpg")

            # Transcribe audio
            transcript = transcribe_audio(audio_data)

            # Generate summary
            summary = generate_summary(transcript)

            # Display results
            st.image(frame, caption="Snapshot")
            st.write("**Transcript:**", transcript)
            st.write(summary)

            # Stop recording if button is pressed
            if st.button("Stop Recording"):
                break

if __name__ == "__main__":
    main()