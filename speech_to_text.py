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
    
    # Hugging Face authentication
    hugging_face_login()

    # Select audio file from 'audios' folder
    audio_path = select_audio_file()

    # Preprocess audio
    waveform, sample_rate = preprocess_audio(audio_path)

    # Debugging: Print the shape of the waveform and sample rate
    print(f"\033[33mWaveform after preprocessing: \033[92m{waveform.shape}")
    print(f"\033[33mSample rate after preprocessing: \033[92m{sample_rate}")

    # Perform transcription and get the text
    transcription = transcribe_audio(waveform, sample_rate)

    # Save transcription to file
    save_transcription(audio_path, transcription)

    # Read text (text-to-speech)
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
    Returns the transcription text.
    """
    print("\n\033[38;2;255;200;189mLoading Wav2Vec2 model and processor...\033[33m")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    # Debugging: Confirm model loading
    print("\033[95mModel and processor successfully loaded!")

    # Add a batch dimension to the waveform
    waveform = waveform.unsqueeze(0)  # Shape becomes [1, sequence_length]
    print(f"\033[33mWaveform shape after adding batch dimension: \033[92m{waveform.shape}")

    # Preprocess the waveform using the processor
    print("\033[38;2;255;200;189mPreprocessing waveform for Wav2Vec2...")
    input_values = processor(
        waveform, 
        sampling_rate=sample_rate, 
        return_tensors="pt", 
        padding="longest"
    ).input_values  # Shape: [1, 1, sequence_length]
    print(f"\033[33mInput values shape before squeezing: \033[92m{input_values.shape}")

    # Remove unnecessary dimensions
    input_values = input_values.squeeze(1)  # Shape becomes [1, sequence_length]
    print(f"\033[33mInput values shape after squeezing: \033[92m{input_values.shape}")

    # Perform inference
    print("\033[38;2;255;200;189mPerforming inference...")
    logits = model(input_values).logits
    print(f"\033[33mLogits shape: \033[92m{logits.shape}")

    # Decode the logits to obtain the transcription
    print("\033[38;2;255;200;189mDecoding logits to text...\033[93mThese action may take a few minutes depending on the length of the audio file")
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    print(f"\n\033[33mTranscription: \033[96m{transcription}")
    return transcription

def save_transcription(audio_path, transcription):
    """
    Save the transcription to a text file in the /transcriptions/ folder.
    The filename will be: <audio_filename>_<YYYYMMDD_HHMMSS>.txt
    """
    transcriptions_dir = os.path.join(os.path.dirname(__file__), "transcriptions")
    os.makedirs(transcriptions_dir, exist_ok=True)
    audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = f"{audio_filename}_{timestamp}.txt"
    txt_path = os.path.join(transcriptions_dir, txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"\033[92mTranscription saved to: \033[94m{txt_path}\033[0m")

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