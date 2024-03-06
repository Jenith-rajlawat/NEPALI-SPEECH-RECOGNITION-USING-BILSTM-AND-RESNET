import os
import random
import pygame
import streamlit as st
import sounddevice as sd
import openai
from scipy.io.wavfile import write
from gtts import gTTS
import webbrowser
from model.configs import UNQ_CHARS
from model.utils import load_model, load_wav, predict_from_wavs
from pydub import AudioSegment
import io
from pydub.playback import play
import keyboard

# Initialize pygame
pygame.init()

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


def play_random_mp3(directory):
    # Get a list of all files in the directory
    files = os.listdir(directory)
    
    # Filter only MP3 files
    mp3_files = [file for file in files if file.endswith('.mp3')]
    
    if mp3_files:
        # Select a random MP3 file
        mp3_file = random.choice(mp3_files)
        
        # Construct the full path to the selected MP3 file
        mp3_path = os.path.join(directory, mp3_file)
        
        # Load and play the selected MP3 file
        pygame.mixer.music.load(mp3_path)
        pygame.mixer.music.play()
        
        # Indicate that the music is playing
        print("Playing:", mp3_file)
    else:
        print("No MP3 files found in the directory.")


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
    duration = 7

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    temp_path = "dataset\\audio\\recorded_audio.wav"
    write(temp_path, fs, recording)
    return temp_path


def play_music(directory):
    """Play a random MP3 file from the specified directory."""
    # Get a list of all files in the directory
    files = os.listdir(directory)
    
    # Filter only MP3 files
    mp3_files = [file for file in files if file.endswith('.mp3')]
    
    if mp3_files:
        # Select a random MP3 file
        mp3_file = random.choice(mp3_files)
        
        # Construct the full path to the selected MP3 file
        mp3_path = os.path.join(directory, mp3_file)
        
        # Load and play the selected MP3 file
        pygame.mixer.music.load(mp3_path)
        pygame.mixer.music.play()

        # Print message
        print(f"Playing: {mp3_file}. Press 's' to stop.")
        
        # Listen for key events to stop the music
        while pygame.mixer.music.get_busy():
            if keyboard.is_pressed('s'):
                # Stop the music when 's' is pressed
                pygame.mixer.music.stop()
                print("Music stopped.")
                break
    else:
        print("No MP3 files found in the directory.")


def perform_transcription(temp_path):
    """Perform transcription and interact with GPT."""
    try:
        import re

        # Inside the try block
        result = transcription_prediction(temp_path)
        extract_text = result[0] if isinstance(result, list) and result else ""
        st.write("User:", extract_text)

        # Check if the text contains 'नमस्ते' at the beginning and 'बजाउ' at the end
        if extract_text.startswith('नमस्ते') and extract_text.endswith('बजाउ'):
            # Specify the directory containing the MP3 files
            directory = r"dataset\music"
            st.write("Playing music. Press 's' to stop.")
            play_music(directory)
            return

        # Check if the text contains 'नमस्ते' and 'गाना खोज' in any order
        elif 'नमस्ते' in extract_text and 'गाना खोज' in extract_text:
            # Extract the search query between "नमस्ते" and "खोज"
            query = re.search(r'नमस्ते(.*?)खोज', extract_text)
            if query:
                search_query = query.group(1).strip()
                youtube_url = "https://www.youtube.com/results?search_query=" + search_query
                webbrowser.open_new_tab(youtube_url)
                st.write("Opening YouTube search for:", search_query)
                return
        
        # Check if the text contains 'नमस्ते' and 'खोज' in any order
        elif 'नमस्ते' in extract_text and 'खोज' in extract_text:
            # Extract the search query between "नमस्ते" and "खोज"
            query = re.search(r'नमस्ते(.*?)खोज', extract_text)
            if query:
                search_query = query.group(1).strip()
                google_url = "https://www.google.com/search?q=" + search_query
                webbrowser.open_new_tab(google_url)
                st.write("Performing Google search for:", search_query)
                return
            
        # with GPT
        myPrompt = "Your answer should be in Nepali. Give me the answer in brief.\n"
        extract_text = myPrompt + "Text: ###\n" + extract_text + "\n###"
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": extract_text}],
        )
        # Display AI response
        st.write("AI Response:", response.choices[0].message.content.strip())
        # Read out the AI response
        text_to_speech(response.choices[0].message.content.strip(), lang='ne')

    except openai.error.OpenAIError as e:
        st.error(f"OpenAI Error: {e}")
    except Exception as e:
        st.error(f"Error during transcription: {e}") 


def main():
    st.title("Nepali Speech Recognition with BiLSTM and ResNet")
    # text_to_speech("नमस्ते मेरो नाम साथि |म तपाइलाई कसरी सहयोग गर्न सक्छु?", lang='ne')

    # Add option for audio source
    audio_source = st.radio("Choose audio source:", ["Upload File", "Record Audio"])

    if audio_source == "Upload File":
        set_openai_api_key()
        # File upload
        audio_file = st.file_uploader("Upload Audio File (flac format)", type=["flac", "wav"])

        if audio_file:
            # Temporarily save the uploaded file to a path
            temp_path = "dataset\\audio\\file.flac"

            with open(temp_path, "wb") as temp_file:
                temp_file.write(audio_file.read())

            # Display the audio file details
            st.audio(temp_path, format="audio/flac")
            st.write(f"File Name: {audio_file.name}")
            st.write(f"File Size: {round(len(audio_file.getvalue()) / 1024, 2)} KB")

            # Perform transcription
            st.subheader("Transcription Result")
            perform_transcription(temp_path)

    elif audio_source == "Record Audio":
        set_openai_api_key()
        if st.button("Start Recording"):
            temp_path = record_audio()
            st.audio(temp_path, format="audio/wav")

            # Perform transcription on the recorded audio
            st.subheader("Transcription Result")
            perform_transcription(temp_path)

if __name__ == "__main__":
    main()
    # streamlit run webapp.py
