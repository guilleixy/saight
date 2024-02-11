import cv2

from ultralytics import YOLO
import supervision as sv
import numpy as np

class Detection:
    def __init__(self, tracker_id, confidence, class_id, class_name):
        self.tracker_id = tracker_id
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.count = 1
        self.persistent = False

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

    # source = 0 for webcam, source = 1 for video input
    # we use a loop to iterate through each frame of the video 
    for result in model.track(source=0, show=False, stream=True):
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            # Update the detections_dict and persistent_detections list
            for _, confidence, class_id, tracker_id in detections:
                class_name = model.model.names[class_id]  
                if tracker_id in detections_dict:
                    detections_dict[tracker_id].count += 1
                    if detections_dict[tracker_id].count > 10 and not detections_dict[tracker_id].persistent:
                        detections_dict[tracker_id].persistent = True
                        persistent_detections.append(detections_dict[tracker_id])
                else:
                    detections_dict[tracker_id] = Detection(tracker_id, confidence, class_id, class_name) 
        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        cv2.imshow("sAIght", frame)

        # Print persistent detections
        print("Persistent detections:")
        for detection in persistent_detections:
            print(detection)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main() 