
import os
import cv2
import numpy as np

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat


# 1. Load ISL dataset
print("Loading ISL dataset...")

ds = load_dataset("Angirekula/isl-isolated-40words")


# 2. Configure MediaPipe
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

landmarker = vision.HandLandmarker.create_from_options(options)


# 3. Target signs
target_words = ["hello", "thank you", "help", "yes", "no"]

output_folder = "extracted_data"
os.makedirs(output_folder, exist_ok=True)

# Timestamp must keep increasing across all videos
timestamp_ms = 0


# 4. Process videos
try:
    for word in target_words:

        print(f"\nProcessing: {word}")

        rows = [
            row for row in ds["train"]
            if row["normalized_word"] == word
        ]

        word_folder = os.path.join(
            output_folder,
            word.replace(" ", "_")
        )

        os.makedirs(word_folder, exist_ok=True)

        for index, row in enumerate(rows):

            cap = None

            try:
                # Download video
                video_path = hf_hub_download(
                    repo_id="Angirekula/isl-isolated-40words",
                    filename=row["video_path"],
                    repo_type="dataset"
                )

                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    print(f"Skipped video {index}: Cannot open video")
                    continue

                landmarks_sequence = []
                frame_index = 0

                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    # Process every third frame
                    if frame_index % 3 == 0:

                        rgb_frame = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2RGB
                        )

                        image = Image(
                            image_format=ImageFormat.SRGB,
                            data=rgb_frame
                        )

                        result = landmarker.detect_for_video(
                            image,
                            timestamp_ms
                        )

                        # Increase timestamp for next processed frame
                        timestamp_ms += 100

                        # Store 21 landmarks for up to 2 hands
                        frame_data = np.zeros(
                            (2, 21, 3),
                            dtype=np.float32
                        )

                        for hand_index, hand in enumerate(
                            result.hand_landmarks[:2]
                        ):

                            for point_index, point in enumerate(hand):

                                frame_data[hand_index, point_index] = [
                                    point.x,
                                    point.y,
                                    point.z
                                ]

                        landmarks_sequence.append(frame_data)

                    frame_index += 1

                cap.release()
                cap = None

                # Save extracted landmarks
                if landmarks_sequence:

                    save_path = os.path.join(
                        word_folder,
                        f"{word.replace(' ', '_')}_{index:03d}.npy"
                    )

                    np.save(
                        save_path,
                        np.array(landmarks_sequence, dtype=np.float32)
                    )

                    print(
                        f"Saved: {save_path} "
                        f"| Frames: {len(landmarks_sequence)}"
                    )

                else:
                    print(f"Skipped video {index}: No frames extracted")

            except Exception as e:
                print(f"Skipped video {index}: {e}")

            finally:
                if cap is not None:
                    cap.release()

finally:
    landmarker.close()

print("\nLandmark extraction completed!")
