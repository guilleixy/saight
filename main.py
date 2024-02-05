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
        detections = sv.Detections.from_yolov8(result)
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        cv2.imshow("sAIghts", frame)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main() 