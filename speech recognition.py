# task2_speech_to_text.py

import speech_recognition as sr

def transcribe_audio(audio_path):
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Load audio file
    with sr.AudioFile(audio_path) as source:
        print("⏳ Listening to audio...")
        audio_data = recognizer.record(source)  # Read the entire audio file

    try:
        print("🧠 Transcribing...")
        # Using Google's free web API for speech recognition
        text = recognizer.recognize_google(audio_data)
        print("\n✅ Transcription:")
        print(text)
    except sr.UnknownValueError:
        print("⚠️ Could not understand the audio.")
    except sr.RequestError as e:
        print(f"⚠️ Could not request results; {e}")

if __name__ == "__main__":
    # Change to your .wav file name
    audio_file = "sample.wav"
    transcribe_audio(audio_file)
