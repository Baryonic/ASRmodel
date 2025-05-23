print("\033[93mRunning speech_to_text.py by \033[96mkeyday electronics")
print("\033[38;2;255;200;189mImporting datasets... please wait...")
import os
from datasets import load_dataset #type: ignore[import]
from huggingface_hub import login #type: ignore[import]
import torchaudio #type: ignore[import]
import torchaudio.transforms as T #type: ignore[import]
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor #type: ignore[import]
import torch #type: ignore[import]
import pyttsx3 #type: ignore[import]
import datetime  # Add this import at the top

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

print("\033[92mdatasets and libraries imported")

def main():
    print("\n\033[38;2;255;200;189mRunning main method...\n")
    
    hugging_face_login()
    audio_path = select_audio_file()
    waveform, sample_rate = preprocess_audio(audio_path)
    print(f"\033[33mWaveform after preprocessing: \033[92m{waveform.shape}")
    print(f"\033[33mSample rate after preprocessing: \033[92m{sample_rate}")

    # Get both transcription and device
    transcription, device = transcribe_audio(waveform, sample_rate)

    # Pass device info to save_transcription
    save_transcription(audio_path, transcription, device)

    read_text(transcription)

def select_audio_file():
    """
    List audio files in the 'audios' folder and let the user select one.
    """
    audios_dir = os.path.join(os.path.dirname(__file__), "audios")
    if not os.path.exists(audios_dir):
        print(f"\033[31mNo 'audios' folder found at {audios_dir}. Please create it and add audio files.\033[0m")
        exit(1)
    audio_files = [f for f in os.listdir(audios_dir) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]
    if not audio_files:
        print(f"\033[31mNo audio files found in {audios_dir}.\033[0m")
        exit(1)
    print("\033[33mAvailable audio files:\033[94m")
    for idx, fname in enumerate(audio_files, 1):
        print(f"  {idx}. {fname}")
    if len(audio_files) == 1:
        print(f"\033[32mOnly one audio file found. Using: \033[94m{audio_files[0]}\033[0m")
        return os.path.join(audios_dir, audio_files[0])
    while True:
        try:
            choice = int(input(f"\n\033[93mSelect the audio file to transcribe (1-{len(audio_files)}):\033[94m"))
            if 1 <= choice <= len(audio_files):
                return os.path.join(audios_dir, audio_files[choice - 1])
            else:
                print("\033[31mInvalid selection. Try again.\033[0m")
        except ValueError:
            print("\033[31mPlease enter a valid number.\033[0m")

def hugging_face_login():
    """
    Authenticate Hugging Face login using an access token from the environment variable.
    """
    print("\033[38;2;255;200;189mAuthenticating with Hugging Face...")
    if not HUGGING_FACE_TOKEN:
        print("\033[31mHUGGING_FACE_TOKEN environment variable not set!\033[0m")
        exit(1)
    login(token=HUGGING_FACE_TOKEN)

def preprocess_audio(audio_path):
    """
    Preprocess the audio: load, resample, and convert to mono.
    """
    print(f"\n\033[38;2;255;200;189mLoading audio file: \033[92m{audio_path}")
    
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"\033[33mOriginal waveform shape: \033[92m{waveform.shape}\033[92m, Sample rate: \033[92m{sample_rate}")

    # Resample the audio to 16 kHz if necessary
    if sample_rate != 16000:
        print("\033[38;2;255;200;189mResampling audio to 16 kHz...")
        waveform = T.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        sample_rate = 16000
    print(f"\033[33mWaveform shape after resampling: \033[92m{waveform.shape}\033[33m, Sample rate: \033[92m{sample_rate}")

    # Convert to mono by averaging channels if the audio is stereo
    if waveform.shape[0] > 1:
        print("\033[38;2;255;200;189mConverting stereo to mono...")
        waveform = waveform.mean(0)  # Shape becomes [sequence_length]

    print(f"\033[33mFinal waveform shape: \033[92m{waveform.shape}\033[33m, Sample rate: \033[92m{sample_rate}")
    return waveform, sample_rate

def transcribe_audio(waveform, sample_rate):
    """
    Perform speech-to-text transcription using Wav2Vec2.
    Returns the transcription text and device used.
    """
    print("\n\033[38;2;255;200;189mLoading Wav2Vec2 model and processor...\033[33m")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    print(f"\033[95mModel and processor successfully loaded! Using device: {device}")

    waveform = waveform.unsqueeze(0)
    input_values = processor(
        waveform, 
        sampling_rate=sample_rate, 
        return_tensors="pt", 
        padding="longest"
    ).input_values
    input_values = input_values.squeeze(1)
    input_values = input_values.to(device)

    print("\033[38;2;255;200;189mPerforming inference...")
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    print(f"\n\033[33mTranscription: \033[96m{transcription}")
    return transcription, str(device)

def save_transcription(audio_path, transcription, device):
    """
    Save the transcription and device info to a text file in the /transcriptions folder.
    """
    transcriptions_dir = os.path.join(os.path.dirname(__file__), "transcriptions")
    os.makedirs(transcriptions_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    output_file = os.path.join(transcriptions_dir, f"{base}_transcription.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Transcription Report\n")
        f.write("===================\n")
        f.write(f"Audio file: {audio_path}\n")
        f.write(f"Device used for processing: {device.upper()}\n")
        f.write("\nTranscription:\n")
        f.write(transcription)
    print(f"\033[32mTranscription and device info saved to {output_file}\033[0m")

def read_text(text):
    """
    Use text-to-speech to read the provided transcription.
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    main()