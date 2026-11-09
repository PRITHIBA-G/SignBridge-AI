
import os
import numpy as np

words = ["hello", "thank_you", "help", "yes", "no"]

total = 0

for word in words:

    folder = os.path.join("cleaned_data", word)

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".npy")
    ] if os.path.exists(folder) else []

    print(f"{word}: {len(files)} videos")

    total += len(files)

print("\nTotal cleaned videos:", total)
