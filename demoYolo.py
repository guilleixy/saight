from ultralytics import YOLO
import cv2

def main():
    # Cargar el modelo YOLOv8
    model = YOLO("yolov8n.pt")

    # source = 0 para webcam, source = 1 para entrada de video
    # usamos un bucle para iterar a través de cada frame del video 
    for result in model.track(source=0, show=True, stream=True, verbose=False):
        frame = result.orig_img
        cv2.imshow("YOLOv8", frame)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()