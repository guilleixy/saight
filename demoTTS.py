import pyttsx3

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Get user input
text = input("Enter the text to speak: ")

# Set the text to be spoken
engine.say(text)

# Speak the text
engine.runAndWait()