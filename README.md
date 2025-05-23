## ASR model - Speech recognition
- move audio files in english into the /audios folder and they will be transcribed
- requires free hugging face token (token should have necessary permisions)
- requires to manually install python libraries
- requires manual training of the Audio Speech Recognition, hugging face's datasets used for this project
- can be trained for different languages multiple datasets are used
## description
- ASR trained with dataset
- audio files are turned into spectograms by torchaudio
- spectograms processed by the ASR that outputs text
- pyttsx3 reads text out loud
## how to run
- install python libraries
- download and import datasets in hugging face's website
- obtain token from website
- paste .wav audio files in the /audios directory
- (Sh) setx HUGGING_FACE_TOKEN "your_hugging_face_token"
- restart cmd
- python speech_to_text.py
## functionality
- versionchecker.py to check versions
- can read .wav files in /audios
- understands in different languages if trained by different datasets
- reads transcription out loud
