import streamlit as st
import sounddevice as sd
import openai
from scipy.io.wavfile import write

from model.configs import UNQ_CHARS
from model.utils import load_model, load_wav, predict_from_wavs

st.config.set_option("server.maxUploadSize", 10)

def set_openai_api_key():
    """Set OpenAI API key."""
    API_KEY = open("API_KEY", "r").read()
    openai.api_key = API_KEY

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

def perform_transcription(temp_path):
    """Perform transcription and interact with GPT."""
    try:
        result = transcription_prediction(temp_path)
        extract_text = result[0] if isinstance(result, list) and result else ""
        extract_text += "?"
        st.write("User:", extract_text)

        # with GPT
        myPrompt = "Your answer should be in nepali. Give me answer in brief.\n"
        # myPrompt = ""
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

    # Add option for audio source
    audio_source = st.radio("Choose audio source:", ["Upload File", "Record Audio"])

    if audio_source == "Upload File":
        set_openai_api_key()
        # File upload
        audio_file = st.file_uploader("Upload Audio File (flac format)", type=["flac"])

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
