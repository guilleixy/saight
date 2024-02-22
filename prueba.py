import speech_recognition as sr

import pyaudio
import audioop
import math

# Crear un reconocedor de voz
r = sr.Recognizer()
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

try:
    # Bucle infinito
    while True:
        # Usar el micrófono como fuente de audio
        with sr.Microphone() as source:
            print("Habla algo:")
            audio = r.listen(source)
            # Leer datos del flujo de audio
            data = stream.read(1024)

            # Calcular el nivel de volumen en decibelios
            rms = audioop.rms(data, 2)  # width=2 for format=paInt16
            db = 20 * math.log10(rms + 1e-6)  # Agregar una pequeña constante para evitar log(0)

            print("Nivel de volumen: {} dB".format(db))
            # if db > 20:
            try:
                # Usar el reconocedor de voz de Google
                text = r.recognize_google(audio, language='es-ES')
                print("Dijiste: {}".format(text))
            except sr.UnknownValueError:
                print("Google Speech Recognition no pudo entender el audio")
            except sr.RequestError as e:
                print("No se pudo solicitar resultados de Google Speech Recognition; {0}".format(e))
            # else: 
                # print("No se detectó audio")
except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    p.terminate()
    # Salir del bucle cuando se presione una tecla
    pass