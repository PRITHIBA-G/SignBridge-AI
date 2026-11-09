
import cv2
import joblib
import numpy as np
import mediapipe as mp

from collections import deque, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

model = joblib.load("signbridge_model.pkl")

print("SignBridge model loaded successfully!")
print("Available signs:", model.classes_)

# ==========================================
# 2. MEDIAPIPE SETUP
# ==========================================

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

# ==========================================
# 3. FRAME AND PREDICTION BUFFERS
# ==========================================

frame_buffer = deque(maxlen=30)
prediction_buffer = deque(maxlen=10)

# ==========================================
# 4. FEATURE EXTRACTION
# ==========================================

def extract_features(frames):

    normalized_frames = []

    for frame in frames:

        normalized_hands = np.zeros((2, 21, 3))

        for hand_index in range(2):

            hand = frame[hand_index]

            if np.all(hand == 0):
                continue

            wrist = hand[0]
            centered = hand - wrist

            distances = np.linalg.norm(
                centered[:, :2],
                axis=1
            )

            scale = np.max(distances)

            if scale > 0:
                centered = centered / scale

            normalized_hands[hand_index] = centered

        normalized_frames.append(
            normalized_hands.flatten()
        )

    normalized_frames = np.array(normalized_frames)

    mean_features = np.mean(normalized_frames, axis=0)
    std_features = np.std(normalized_frames, axis=0)

    return np.concatenate([
        mean_features,
        std_features
    ])


# ==========================================
# 5. WEBCAM SETUP
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
    raise SystemExit

print("\nSignBridge ISL Recognition Started!")
print("Show a sign to the camera.")
print("Press Q to quit.")

timestamp_ms = 0

prediction_text = "Show your hand"
confidence_text = ""

last_debug_prediction = None

# ==========================================
# 6. START RECOGNITION
# ==========================================

try:

    with vision.HandLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            timestamp_ms += 33

            current_frame = np.zeros((2, 21, 3))

            # ==================================
            # 7. HAND LANDMARK DETECTION
            # ==================================

            if result.hand_landmarks:

                for hand_index, hand_landmarks in enumerate(
                    result.hand_landmarks[:2]
                ):

                    for landmark_index, landmark in enumerate(
                        hand_landmarks
                    ):

                        current_frame[
                            hand_index,
                            landmark_index
                        ] = [
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ]

                frame_buffer.append(current_frame)

                # ==================================
                # 8. MODEL PREDICTION
                # ==================================

                if len(frame_buffer) == 30:

                    features = extract_features(
                        list(frame_buffer)
                    )

                    features = features.reshape(1, -1)

                    prediction = model.predict(features)[0]

                    probabilities = model.predict_proba(features)[0]

                    confidence = float(
                        np.max(probabilities) * 100
                    )

                    prediction_buffer.append(
                        (prediction, confidence)
                    )

                    # ==================================
                    # 9. TERMINAL DIAGNOSTICS
                    # ==================================

                    if prediction != last_debug_prediction:

                        print("\n----------------------------")
                        print("Current prediction:", prediction)

                        print("Class probabilities:")

                        for label, probability in zip(
                            model.classes_,
                            probabilities
                        ):

                            print(
                                f"{label}: "
                                f"{probability * 100:.2f}%"
                            )

                        print("----------------------------")

                        last_debug_prediction = prediction

                    # ==================================
                    # 10. SMOOTH PREDICTIONS
                    # ==================================

                    labels = [
                        item[0]
                        for item in prediction_buffer
                    ]

                    counts = Counter(labels)

                    best_label, votes = counts.most_common(1)[0]

                    if votes >= 6:

                        prediction_text = best_label.replace(
                            "_", " "
                        )

                        matching_confidences = [
                            conf
                            for label, conf in prediction_buffer
                            if label == best_label
                        ]

                        average_confidence = np.mean(
                            matching_confidences
                        )

                        confidence_text = (
                            f"Model confidence: "
                            f"{average_confidence:.1f}%"
                        )

                    else:

                        prediction_text = "Uncertain"

                        confidence_text = (
                            "Waiting for stable prediction"
                        )

            else:

                frame_buffer.clear()
                prediction_buffer.clear()

                prediction_text = "Show your hand"
                confidence_text = ""

                last_debug_prediction = None

            # ==================================
            # 11. DISPLAY TOP THREE PREDICTIONS
            # ==================================

            top_indices = []

            if len(frame_buffer) == 30:

                top_indices = np.argsort(
                    probabilities
                )[::-1][:3]

            # Display predicted sign
            cv2.putText(
                frame,
                "Sign: " + prediction_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                confidence_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Frames: {len(frame_buffer)}/30",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            if prediction_buffer:

                stable_votes = Counter(
                    item[0] for item in prediction_buffer
                ).most_common(1)[0][1]

                cv2.putText(
                    frame,
                    f"Stable predictions: {stable_votes}/10",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            # Display top three classes
            for rank, index in enumerate(top_indices):

                label = model.classes_[index]

                probability = probabilities[index] * 100

                cv2.putText(
                    frame,
                    f"{rank + 1}. {label}: {probability:.1f}%",
                    (20, 205 + rank * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

            cv2.imshow(
                "SignBridge - ISL Recognition",
                frame
            )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:

    camera.release()
    cv2.destroyAllWindows()

print("\nSign recognition stopped.")
