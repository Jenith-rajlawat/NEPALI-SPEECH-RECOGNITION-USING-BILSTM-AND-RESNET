import sounddevice as sd
import soundfile as sf
import os


def record_and_save_audio(file_path, duration=5, samplerate=44100):
    print("Recording...")

    # Record audio
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=2, dtype='int16')
    sd.wait()

    # Save audio as .flac file
    sf.write(file_path, audio_data, samplerate)

    print(f"Audio recorded and saved to: {file_path}")


if __name__ == "__main__":
    # Set the directory path where you want to save the audio file
    output_directory = r"./dataset"

    # Create a subdirectory named "audio" within the output directory
    audio_directory = os.path.join(output_directory, "audio")
    os.makedirs(audio_directory, exist_ok=True)

 # Specify the file name for the recorded audio
    output_file_name = "recorded_audio.flac"
    output_file_path = os.path.join(audio_directory, output_file_name)

    # Specify the duration of the recording in seconds
    recording_duration = 3

    # Record and save audio
    record_and_save_audio(output_file_path, duration=recording_duration)