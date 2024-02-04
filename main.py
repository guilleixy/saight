from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    ## source = 0 for webcam, source = 1 for video input
    result = model.track(source=0, show=True)

if __name__ == "__main__":
    main() 