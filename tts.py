import pyttsx3

# Initialize engine
engine = pyttsx3.init()

# Set speaking speed
engine.setProperty('rate', 150)

# Set volume
engine.setProperty('volume', 1.0)

# Take text input
text = input("Enter text: ")

# Convert text to speech
engine.say(text)

# Save speech to audio file
engine.save_to_file(text, "output.wav")

# Run engine
engine.runAndWait()

print("Done! Audio saved as output.wav")
