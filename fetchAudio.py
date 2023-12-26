import sounddevice as sd
import soundfile as sf
import os
import numpy as np


def record_and_save_audio(file_path, duration=5, samplerate=44100, silence_threshold=0.01):
    print("Recording...")

    # Set some parameters for better audio quality
    device_info = sd.query_devices(sd.default.device, 'input')
    channels = min(device_info['max_input_channels'], 2)

    # Record audio with higher precision (float32)
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=channels, dtype='float32')
    sd.wait()

    # Normalize the audio to the range [-1, 1]
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Find the index where the absolute values exceed the silence threshold
    non_silent_indices = np.where(np.abs(audio_data) > silence_threshold)[0]

    # Determine the start and end indices for non-silent audio
    start_idx = non_silent_indices[0] if len(non_silent_indices) > 0 else 0
    end_idx = non_silent_indices[-1] + 1 if len(non_silent_indices) > 0 else len(audio_data)

    # Trim the audio
    audio_data_trimmed = audio_data[start_idx:end_idx]

    # Save trimmed audio as .flac file
    sf.write(file_path, audio_data_trimmed, samplerate)

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
    recording_duration = 5

    # Record and save audio with trimming
    record_and_save_audio(output_file_path, duration=recording_duration)
