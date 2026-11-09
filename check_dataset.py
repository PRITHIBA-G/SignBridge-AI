
import os
import numpy as np

dataset_folder = "extracted_data"

total_files = 0
total_valid_frames = 0
total_frames = 0

for word in ["hello", "thank_you", "help", "yes", "no"]:

    folder = os.path.join(dataset_folder, word)

    print(f"\n--- {word.upper()} ---")

    if not os.path.exists(folder):
        print("Folder not found!")
        continue

    files = [f for f in os.listdir(folder) if f.endswith(".npy")]

    valid_files = 0

    for file in files:

        path = os.path.join(folder, file)
        data = np.load(path)

        # Check whether at least one hand was detected
        detected_frames = np.any(data[:, :, :, :2] != 0, axis=(1, 2, 3))

        valid_count = np.sum(detected_frames)

        total_files += 1
        total_frames += len(data)
        total_valid_frames += valid_count

        if valid_count > 0:
            valid_files += 1

        print(
            f"{file}: Shape={data.shape}, "
            f"Detected frames={valid_count}/{len(data)}"
        )

    print(f"Videos with detected hands: {valid_files}/{len(files)}")

print("\n--- OVERALL SUMMARY ---")
print("Total files:", total_files)
print("Total frames:", total_frames)
print("Frames with detected hands:", total_valid_frames)

if total_frames > 0:
    percentage = (total_valid_frames / total_frames) * 100
    print(f"Hand detection percentage: {percentage:.2f}%")
