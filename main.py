import cv2
import pyttsx3

from ultralytics import YOLO
import supervision as sv
import numpy as np

import threading
import queue

import speech_recognition as sr

import openai
from dotenv import load_dotenv
import os

import boto3
import csv
import time

import copy

with open('demoCredentials.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        access_key_id = row[0]
        secret_access_key = row[1]

client = boto3.client('rekognition', region_name='eu-west-2', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

speech_queue = queue.Queue()
frame_queue = queue.Queue()

# Crear un reconocedor de voz
r = sr.Recognizer()

# A list of words similar to saigth to activate the voice assistant
saigth_words = ["saigth", "site", "sight", "cite", "strike", "inside"]
gpt_words = ["GPT", "gpt", "Gpt", "gepete", "jepete"]
images_path = "outputimages/"


def save_frame(frame, filename, folder):
    # Create the full path for the output file
    path = f"{folder}/{filename}"
    # Save the frame as an image file
    cv2.imwrite(path, frame)   

def rekognition_response(photo):
    with open(photo, 'rb') as source_image:
        source_bytes = source_image.read()

    detect_objects = client.detect_labels(Image={'Bytes': source_bytes})
    for label in detect_objects['Labels']:
        print(label['Name'] + ' : ' + str(label['Confidence']))
    

def text_rekognition_response(photo):
    with open(photo, 'rb') as source_image:
        source_bytes = source_image.read()

    detect_text = client.detect_text(Image={'Bytes': source_bytes})
    for label in detect_text['TextDetections']:
        print(label['DetectedText'] + ' : ' + str(label['Confidence']))
    speech_queue.put(detect_text['TextDetections']) 
# estoy hay que arreglarlo
def facial_rekognition_response(photo):
    with open(photo, 'rb') as source_image:
        source_bytes = source_image.read()

    detect_faces = client.detect_faces(Image={'Bytes': source_bytes})
    for label in detect_faces['FaceDetails']:
        print(label['Emotions'])
    speech_queue.put(detect_faces['Emotions'])

def famouse_rekognition_response(photo):
    with open(photo, 'rb') as source_image:
        source_bytes = source_image.read()
    detect_famous = client.recognize_celebrities(Image={'Bytes': source_bytes})
    for celebrity in detect_famous['CelebrityFaces']:
        speech_queue.put(celebrity['Name'])

def chatgpt_response(prompt):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        temperature=0.1,
        messages=[
            {"role": "system", "content": "Eres parte de una aplicación destinada a ayudar a personas con discapacidad visual a interactuar de forma más efectiva con su entorno. En concreto eres su asistente personal, encargado de recibir la información del entorno y ayudar al usuario con estos datos de la forma que él te solicite. A continuación, recibes una lista de elementos que hay en el entorno. Ten en cuenta que algunos pueden no estar relacionados con lo que te preguntan."},
            {"role": "user", "content": prompt}
        ]
    )
    speech_queue.put(response.choices[0].message['content'])

def speak(engine, queue):
    while True:
        text = queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

def listen(frame_queue, persistent_detections):
    while True:
        with sr.Microphone() as source:
            print("Habla algo:")
            audio = r.listen(source)
            try:
                # Usar el reconocedor de voz de Google
                text = r.recognize_google(audio, language='es-ES')

                # Comandos de voz
                if any(word in text for word in gpt_words):
                    speech_queue.put(chatgpt_response(prompt = text + persistent_detections))
                if any(word in text for word in saigth_words):
                    speech_queue.put("Esto lo respondería Saight")
                    if ("profundo" or "detallado" in text):
                        # Save the frame as an image file
                        frame = frame_queue.get()
                        filename = f"output_{int(time.time())}.jpg"
                        save_frame(frame, filename, "outputimages")
                        rekognition_response(images_path + filename)
                    elif ("texto" or "escrito" or "pone" in text):
                        frame = frame_queue.get()
                        filename = f"output_{int(time.time())}.jpg"
                        save_frame(frame, filename, "outputimages") 
                        text_rekognition_response(images_path + filename)                       
                    elif("facial" or "emocion" in text):
                        frame = frame_queue.get()
                        filename = f"output_{int(time.time())}.jpg"
                        save_frame(frame, filename, "outputimages")                        
                        facial_rekognition_response(images_path + filename)

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
    listen_thread = threading.Thread(target=listen, args=(frame_queue, persistent_detections))
    listen_thread.start()

    # source = 0 for webcam, source = 1 for video input
    # we use a loop to iterate through each frame of the video 
    for result in model.track(source=0, show=False, stream=True, verbose=False):
        frame = result.orig_img
        frame_without_boxes = copy.deepcopy(frame)  # Create a copy of the frame before boxes are drawn
        frame_queue.put(frame_without_boxes) 
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