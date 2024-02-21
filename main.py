import cv2
import pyttsx3

from ultralytics import YOLO
import supervision as sv
import numpy as np

import threading
import queue

import speech_recognition as sr

speech_queue = queue.Queue()

# Crear un reconocedor de voz
r = sr.Recognizer()

def speak(engine, queue):
    while True:
        text = queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

def listen():
    while True:
        print("Escuchando...")
        with sr.Microphone() as source:
            print("Habla algo:")
            audio = r.listen(source)
            try:
                # Usar el reconocedor de voz de Google
                text = r.recognize_google(audio, language='es-ES')
                if "hola" in text:
                    speech_queue.put("Hola, ¿en qué puedo ayudarte?")
                print("Dijiste: {}".format(text))
            except sr.UnknownValueError:
                print("Google Speech Recognition no pudo entender el audio")
            except sr.RequestError as e:
                print("No se pudo solicitar resultados de Google Speech Recognition; {0}".format(e))

class Detection:
    def __init__(self, tracker_id, confidence, class_id, class_name):
        self.tracker_id = tracker_id
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.count = 1
        self.persistent = False
        self.last_seen = 0

    def __str__(self):
        return f"Tracker ID: {self.tracker_id}, Class ID: {self.class_id}, Class Name: {self.class_name}, Confidence: {self.confidence}, Count: {self.count}"  

def main():
    model = YOLO("yolov8n.pt")
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
    )

    detections_dict = {}
    persistent_detections = []

    engine = pyttsx3.init()
    engine.setProperty('voice', 'en_US')
    engine.setProperty("rate", 125)

    speech_thread = threading.Thread(target=speak, args=(engine, speech_queue))
    speech_thread.start()
    listen_thread = threading.Thread(target=listen)
    listen_thread.start()

    # source = 0 for webcam, source = 1 for video input
    # we use a loop to iterate through each frame of the video 
    for result in model.track(source=0, show=False, stream=True, verbose=False):
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            # Update the detections_dict and persistent_detections list
            # Track ID works a bit strange. As a short cut, the class_id is used to track persistent objects as for now
            # the number of objects does not matter.
            
            #filter detections with confidence > 0.5
            detections = detections[detections.confidence > 0.5]
            for _, confidence, class_id, tracker_id in detections:
                class_name = model.model.names[class_id]  
                if class_id in detections_dict:
                    detections_dict[class_id].count += 1
                    detections_dict[class_id].last_seen = 0
                    # print(f"Class ID: {class_id}, Count: {detections_dict[class_id].count}, Name: {detections_dict[class_id].class_name}")
                    if detections_dict[class_id].count > 5 and not detections_dict[class_id].persistent:
                        detections_dict[class_id].persistent = True
                        persistent_detections.append(detections_dict[class_id])
                        speech_queue.put(f"{detections_dict[class_id].class_name} detected")
                else:
                    detections_dict[class_id] = Detection(tracker_id, confidence, class_id, class_name)
            # Increment last_seen for each persistent detection and remove it if it hasn't been seen for 80 frames
            for class_id, detection in list(detections_dict.items()):  # Use list to avoid RuntimeError
                if detection.persistent:
                    detection.last_seen += 1
                    if detection.last_seen >= 60:
                        del detections_dict[class_id]
                        persistent_detections.remove(detection)

        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        cv2.imshow("sAIght", frame)

        # Print persistent detections
        # print("Persistent detections:")
        # for detection in persistent_detections:
        #     print(detection)

        if (cv2.waitKey(30) == 27):
            break
    speech_queue.put(None)
    speech_thread.join()

if __name__ == "__main__":
    main() 