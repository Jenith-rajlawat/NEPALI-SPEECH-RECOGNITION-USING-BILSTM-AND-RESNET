import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write

from model.configs import UNQ_CHARS
from model.utils import CER_from_wavs, ctc_softmax_output_from_wavs, load_model, load_wav, plot_losses, predict_from_wavs

def transcription_prediction(wav_path):
    """Predicts and returns the transcription for the given audio file."""
    # Load the trained model
    model = load_model("model/trained_model.h5")

    # Load the wav file
    wav = load_wav(wav_path)

    # Predict transcription
    sentences, char_indices = predict_from_wavs(model, [wav], UNQ_CHARS)

    return sentences[0]

def record_audio():
    """Records audio from the microphone and saves it to a temporary WAV file."""
    fs = 16000  # Sample rate
    duration = 5  # Recording duration in seconds

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished

    temp_path = "dataset\\audio\\recorded_audio.wav"
    write(temp_path, fs, recording)  # Write the recording to a WAV file

    return temp_path

def main():
    st.title("Nepali Speech Recognition with BiLSTM and ResNet")

    # Add option for audio source
    audio_source = st.radio("Choose audio source:", ["Upload File", "Record Audio"])

    if audio_source == "Upload File":
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
            try:
                result = transcription_prediction(temp_path)
                st.success(result)
            except Exception as e:
                st.error(f"Error during transcription: {e}")

    elif audio_source == "Record Audio":
        if st.button("Start Recording"):
            temp_path = record_audio()
            st.audio(temp_path, format="audio/wav")

            # Perform transcription on the recorded audio
            st.subheader("Transcription Result")
            try:
                result = transcription_prediction(temp_path)
                st.success(result)
            except Exception as e:
                st.error(f"Error during transcription: {e}")

if __name__ == "__main__":
    main()
    # streamlit run "C:\PARA\--Project--\NEPALI-SPEECH-RECOGNITION-USING-BILSTM-AND-RESNET\webapp.py"
