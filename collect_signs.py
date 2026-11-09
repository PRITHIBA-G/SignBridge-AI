
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"
DATASET_DIR = BASE_DIR / "dataset"

# Sign labels and keyboard numbers
labels = {
    "1": "hello",
    "2": "thank_you",
    "3": "help",
    "4": "yes",
    "5": "no"
}

# Collect 30 frames for each sample
SEQUENCE_LENGTH = 30

# MediaPipe setup
base_options = mp.tasks.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = mp.tasks.vision.HandLandmarker.create_from_options(
    options
)

# Webcam setup
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access webcam!")
    detector.close()
    raise SystemExit

selected_label = None
recording = False
sequence = []

# Hand connections for drawing
connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

print("SignBridge AI - Dataset Collection")
print("1 = Hello | 2 = Thank You | 3 = Help")
print("4 = Yes | 5 = No")
print("S = Start recording | Q = Quit")

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Camera error!")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        landmarks_data = None

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            height, width, _ = frame.shape
            points = []

            # Extract 21 hand landmarks
            landmarks_data = []

            for landmark in hand:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

                landmarks_data.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

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
                frame, "Hand Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

        else:
            cv2.putText(
                frame, "Show your hand",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2
            )

        # Collect frames while recording
        if recording and landmarks_data is not None:
            sequence.append(landmarks_data)

            if len(sequence) == SEQUENCE_LENGTH:
                folder = DATASET_DIR / selected_label
                folder.mkdir(parents=True, exist_ok=True)

                existing = list(folder.glob("*.npy"))
                sample_number = len(existing) + 1

                filename = folder / (
                    f"{selected_label}_{sample_number:03d}.npy"
                )

                np.save(filename, np.array(sequence))

                print(f"Saved: {filename.name}")

                sequence = []
                recording = False

        # Display instructions
        current = selected_label if selected_label else "Not selected"

        cv2.putText(
            frame, f"Selected: {current}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )

        status = "RECORDING" if recording else "READY"

        cv2.putText(
            frame, status,
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2
        )

        cv2.imshow("SignBridge - Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        # Select sign
        if chr(key) in labels:
            if not recording:
                selected_label = labels[chr(key)]
                sequence = []
                print("Selected:", selected_label)

        # Start recording
        elif key == ord("s"):
            if not recording and selected_label:
                sequence = []
                recording = True
                print("Recording:", selected_label)

            elif not selected_label:
                print("Select a sign first!")

        # Cancel current sample
        elif key == ord("c"):
            recording = False
            sequence = []
            print("Sample cancelled.")

        # Quit
        elif key == ord("q"):
            break

finally:
    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("Dataset collection stopped.")
