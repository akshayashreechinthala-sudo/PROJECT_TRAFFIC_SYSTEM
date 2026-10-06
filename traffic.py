import cv2
from ultralytics import YOLO
import time

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open laptop webcam
cap = cap = cv2.VideoCapture("traffic.mp4")

# COCO vehicle classes
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
vehicle_classes = [2, 3, 5, 7]

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not found!")
        break

    # Detect objects
    results = model(frame, verbose=False)

    vehicle_count = 0

    # Draw detections
    for result in results:

        boxes = result.boxes

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id in vehicle_classes and confidence > 0.5:

                vehicle_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Vehicle label
                label = model.names[class_id]

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # Calculate traffic signal time
    if vehicle_count <= 5:

        green_time = 20
        traffic_level = "LOW"

    elif vehicle_count <= 15:

        green_time = 40
        traffic_level = "MEDIUM"

    else:

        green_time = 60
        traffic_level = "HIGH"

    # Display information
    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Traffic: {traffic_level}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        
    )

    cv2.putText(
        frame,
        f"Green Time: {green_time} seconds",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # Display camera
    cv2.imshow(
        "AI Smart Traffic Management System",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
cap.release()
cv2.destroyAllWindows()
