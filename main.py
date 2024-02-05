import cv2

from ultralytics import YOLO
import supervision as sv

def main():
    model = YOLO("yolov8n.pt")
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
    )
    # source = 0 for webcam, source = 1 for video input
    # we use a loop to iterate through each frame of the video 
    for result in model.track(source=0, show=False, stream=True):
        frame = result.orig_img
        detections = sv.Detections.from_ultralytics(result)
        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for tracker_id, confidence, class_id in zip(
                detections.tracker_id, detections.confidence, detections.class_id
            )
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        cv2.imshow("sAIghts", frame)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main() 