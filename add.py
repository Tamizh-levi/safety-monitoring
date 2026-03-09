import cv2
from ultralytics import YOLO

# 1. Load the model (YOLO26 is natively NMS-free for lower latency)
model = YOLO("yolo26n.pt")

# 2. Initialize the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Set resolution (optional, helps performance)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting Real-Time Detection... Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    # 3. Run inference on the current frame
    # classes=[0] filters for 'person'
    # stream=True is more memory efficient for video
    results = model.predict(source=frame, classes=[0], conf=0.5, stream=True)

    for r in results:
        # Plot the detections directly onto the frame
        annotated_frame = r.plot()

        # 4. Display the frame
        cv2.imshow("YOLO26 Real-Time Person Detection", annotated_frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()