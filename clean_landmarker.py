
import os
import numpy as np

source_folder = "extracted_data"
output_folder = "cleaned_data"

target_words = ["hello", "thank_you", "help", "yes", "no"]

os.makedirs(output_folder, exist_ok=True)

MIN_DETECTION_PERCENTAGE = 40

total_saved = 0
total_skipped = 0

for word in target_words:

    print(f"\nProcessing: {word}")

    input_path = os.path.join(source_folder, word)
    output_path = os.path.join(output_folder, word)

    os.makedirs(output_path, exist_ok=True)

    if not os.path.exists(input_path):
        print("Source folder not found!")
        continue

    files = [f for f in os.listdir(input_path) if f.endswith(".npy")]

    for file in files:

        data = np.load(os.path.join(input_path, file))

        # Check whether at least one hand is detected
        detected_frames = np.any(
            data[:, :, :, :2] != 0,
            axis=(1, 2, 3)
        )

        detection_percentage = (
            np.sum(detected_frames) / len(data)
        ) * 100

        # Skip videos below 40% detection
        if detection_percentage < MIN_DETECTION_PERCENTAGE:
            print(f"Skipped: {file} ({detection_percentage:.1f}%)")
            total_skipped += 1
            continue

        # Keep only frames with detected hands
        cleaned_data = data[detected_frames]

        save_path = os.path.join(output_path, file)

        np.save(save_path, cleaned_data)

        print(
            f"Saved: {file} | "
            f"Detection: {detection_percentage:.1f}% | "
            f"Frames: {len(cleaned_data)}"
        )

        total_saved += 1

print("\n========== CLEANING SUMMARY ==========")
print("Videos saved:", total_saved)
print("Videos skipped:", total_skipped)
print("Total processed:", total_saved + total_skipped)
print("Cleaned dataset location:", output_folder)
