
import os
import numpy as np

WORDS = ["hello", "thank_you", "help", "yes", "no"]

X = []
y = []

def extract_features(data):
    normalized_frames = []

    for frame in data:
        normalized_hands = np.zeros((2, 21, 3))

        for hand_index in range(2):
            hand = frame[hand_index]

            # Skip a hand if no landmarks were detected
            if np.all(hand == 0):
                continue

            # Use wrist landmark as the reference point
            wrist = hand[0]
            centered = hand - wrist

            # Normalize hand size
            distances = np.linalg.norm(centered[:, :2], axis=1)
            scale = np.max(distances)

            if scale > 0:
                centered = centered / scale

            normalized_hands[hand_index] = centered

        normalized_frames.append(normalized_hands.flatten())

    normalized_frames = np.array(normalized_frames)

    # Calculate mean and standard deviation
    mean_features = np.mean(normalized_frames, axis=0)
    std_features = np.std(normalized_frames, axis=0)

    return np.concatenate([mean_features, std_features])


# Read cleaned videos
for word in WORDS:
    folder = os.path.join("cleaned_data", word)

    for filename in os.listdir(folder):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder, filename)

            data = np.load(file_path)

            features = extract_features(data)

            X.append(features)
            y.append(word)

            print("Processed:", word, filename)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Save training data
np.savez(
    "training_data.npz",
    X=X,
    y=y
)

print("\nTraining data prepared successfully!")
print("Feature shape:", X.shape)
print("Labels shape:", y.shape)
print("Total videos:", len(y))
print("Classes:", np.unique(y))

