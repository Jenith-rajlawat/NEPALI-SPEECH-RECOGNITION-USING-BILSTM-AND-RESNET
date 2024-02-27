import os
import random
import pygame
import streamlit as st
import sounddevice as sd
import openai
from scipy.io.wavfile import write
from gtts import gTTS
import webbrowser
from pydub import AudioSegment
import io
from pydub.playback import play
import re
import keyboard

# Initialize pygame
pygame.init()

# Set maximum upload size
st.config.set_option("server.maxUploadSize", 10)

def set_openai_api_key():
    """Set OpenAI API key."""
    API_KEY = open("API_KEY", "r").read()
    openai.api_key = API_KEY

def text_to_speech(text, lang='ne'):
    """Convert text to speech and play on the website."""
    try:
        tts = gTTS(text=text, lang=lang)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        audio_segment = AudioSegment.from_file(audio_bytes)
        play(audio_segment)
    except Exception as e:
        st.error("Error during text-to-speech conversion: " + str(e))

def transcription_prediction(wav_path):
    """Predicts and returns the transcription for the given audio file."""
    model = load_model("model/trained_model.h5")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    wav = load_wav(wav_path)
    sentences = predict_from_wavs(model, [wav], UNQ_CHARS)
    return sentences[0]

def record_audio():
    """Records audio from the microphone and saves it to a temporary WAV file."""
    fs = 16000
    duration = 5

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    temp_path = "dataset\\audio\\recorded_audio.wav"
    write(temp_path, fs, recording)
    return temp_path

def play_music(directory):
    """Play a random MP3 file from the specified directory."""
    files = os.listdir(directory)
    mp3_files = [file for file in files if file.endswith('.mp3')]
    
    if mp3_files:
        mp3_file = random.choice(mp3_files)
        mp3_path = os.path.join(directory, mp3_file)
        pygame.mixer.music.load(mp3_path)
        pygame.mixer.music.play()
        print(f"Playing: {mp3_file}. Press 's' to stop.")
        
        while pygame.mixer.music.get_busy():
            if keyboard.is_pressed('s'):
                pygame.mixer.music.stop()
                print("Music stopped.")
                break
    else:
        print("No MP3 files found in the directory.")

def perform_transcription(temp_path):
    """Perform transcription and interact with GPT."""
    try:
        import re

        result = transcription_prediction(temp_path)
        extract_text = result[0] if isinstance(result, list) and result else ""
        st.write("User:", extract_text)

        if extract_text.startswith('नमस्ते') and extract_text.endswith('बजाउ'):
            directory = r"C:\Users\dell\Desktop\NEPALI-SPEECH-RECOGNITION-USING-BILSTM-AND-RESNET\dataset\music"
            st.write("Playing music. Press 's' to stop.")
            play_music(directory)
            return

        elif 'नमस्ते' in extract_text and 'बजाउ' in extract_text:
            query = re.search(r'नमस्ते(.*?)बजाउ', extract_text)
            if query:
                search_query = query.group(1).strip()
                youtube_url = "https://www.youtube.com/results?search_query=" + search_query
                webbrowser.open_new_tab(youtube_url)
                st.write("Opening YouTube search for:", search_query)
                return
        
        elif 'नमस्ते' in extract_text and 'खोज' in extract_text:
            query = re.search(r'नमस्ते(.*?)खोज', extract_text)
            if query:
                search_query = query.group(1).strip()
                google_url = "https://www.google.com/search?q=" + search_query
                webbrowser.open_new_tab(google_url)
                st.write("Performing Google search for:", search_query)
                return

        myPrompt = "Your answer should be in Nepali. Give me the answer in paragraph form.\n"
        extract_text = myPrompt + "Text: ###\n" + extract_text + "\n###"
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": extract_text}],
        )
        st.write("AI Response:", response.choices[0].message.content.strip())

    except openai.error.OpenAIError as e:
        st.error(f"OpenAI Error: {e}")
    except Exception as e:
        st.error(f"Error during transcription: {e}") 

def main():
    st.title("Nepali Speech Recognition with BiLSTM and ResNet")

    audio_source = st.radio("Choose audio source:", ["Upload File", "Record Audio"])

    if audio_source == "Upload File":
        set_openai_api_key()
        audio_file = st.file_uploader("Upload Audio File (flac format)", type=["flac", "wav"])

        if audio_file:
            temp_path = "dataset\\audio\\file.flac"

            with open(temp_path, "wb") as temp_file:
                temp_file.write(audio_file.read())

            st.audio(temp_path, format="audio/flac")
            st.write(f"File Name: {audio_file.name}")
            st.write(f"File Size: {round(len(audio_file.getvalue()) / 1024, 2)} KB")

            st.subheader("Transcription Result")
            perform_transcription(temp_path)

    elif audio_source == "Record Audio":
        set_openai_api_key()
        if st.button("Start Recording"):
            temp_path = record_audio()
            st.audio(temp_path, format="audio/wav")
            st.subheader("Transcription Result")
            perform_transcription(temp_path)

if __name__ == "__main__":
    main()
