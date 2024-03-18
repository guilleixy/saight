import speech_recognition as sr

# Create a recognizer object
r = sr.Recognizer()

# Use the default microphone as the audio source
with sr.Microphone() as source:
    print("Escuchando...")

    # Listen for audio and convert it to text
    audio = r.listen(source)

    try:
        # Use Google Speech Recognition to recognize the audio
        text = r.recognize_google(audio)
        print("Has dicho:", text)
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand.")
    except sr.RequestError as e:
        print("Sorry, an error occurred:", str(e))