import os
import time
import logging
from datetime import datetime
from ultralytics import YOLO
import cv2


logging.basicConfig(filename='object_detection.log', level=logging.INFO)

SAVE_PATH = 'detected_objects'
MAX_FOLDER_SIZE = 4 * 1024 * 1024
IMAGE_LIFETIME = 3 * 60


if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def get_folder_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def delete_old_images():
    current_time = time.time()
    for filename in os.listdir(SAVE_PATH):
        file_path = os.path.join(SAVE_PATH, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > IMAGE_LIFETIME:
                os.remove(file_path)
                logging.info(f"Deleted {filename} after {IMAGE_LIFETIME} seconds")

def delete_images_by_size():
    while get_folder_size(SAVE_PATH) > MAX_FOLDER_SIZE:
        oldest_file = min(
            (os.path.join(SAVE_PATH, f) for f in os.listdir(SAVE_PATH)),
            key=os.path.getctime
        )
        os.remove(oldest_file)
        logging.info(f"Deleted {oldest_file} to keep folder size below {MAX_FOLDER_SIZE} bytes")


rtsp_link = 'rtsp://admin:pass@123@192.168.1.240:554/cam/realmonitor?channel=3&subtype=0'
cap = cv2.VideoCapture(rtsp_link)


model = YOLO('yolov8n.pt')

while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model(frame)
    detections = results[0].boxes

    for det in detections:
        bbox = det.xyxy[0].cpu().numpy().astype(int)
        confidence = det.conf.cpu().numpy()

        if confidence > 0.5:
            startX, startY, endX, endY = bbox


            object_img = frame[startY:endY, startX:endX]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{timestamp}.jpg"
            filepath = os.path.join(SAVE_PATH, filename)
            cv2.imwrite(filepath, object_img)
            logging.info(f"Detected object saved as {filename}")


    delete_old_images()
    delete_images_by_size()


    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
