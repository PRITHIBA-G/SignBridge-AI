
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model file
MODEL_PATH = "hand_landmarker.task"

# Load MediaPipe model
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

# Start webcam
cap = cv2.VideoCapture(0)

print("SignBridge AI - Hand Detection")
print("Press Q to close")

connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Camera not working")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            height, width, _ = frame.shape
            points = []

            for landmark in hand:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

                cv2.circle(
                    frame, (x, y), 5,
                    (0, 255, 0), -1
                )

            for start, end in connections:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 0, 0),
                    2
                )

        cv2.putText(
            frame,
            "Hand Detected!",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:
        cv2.putText(
            frame,
            "Show your hand",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("SignBridge AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
