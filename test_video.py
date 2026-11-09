from datasets import load_dataset
from huggingface_hub import hf_hub_download
import cv2
import os

ds = load_dataset("Angirekula/isl-isolated-40words")

row = next(
    row for row in ds["train"]
    if row["normalized_word"] == "hello"
)

video_path = hf_hub_download(
    repo_id="Angirekula/isl-isolated-40words",
    filename=row["video_path"],
    repo_type="dataset"
)

cap = cv2.VideoCapture(video_path)

print("Video downloaded:", os.path.exists(video_path))
print("Video opened:", cap.isOpened())
print("Frame count:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

cap.release()