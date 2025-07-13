import os
import shutil
import random

base_dir = "Dataset"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

output_dirs = {
    "train": 0.8,
    "val": 0.2,
}

for split in output_dirs:
    os.makedirs(os.path.join(base_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, "labels"), exist_ok=True)

image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)

total = len(image_files)
start = 0
for split, ratio in output_dirs.items():
    count = int(total * ratio)
    selected_files = image_files[start:start + count]
    for file in selected_files:
        src_img = os.path.join(images_dir, file)
        dst_img = os.path.join(base_dir, split, "images", file)
        shutil.copy(src_img, dst_img)

        label_file = file.rsplit('.', 1)[0] + ".txt"
        src_label = os.path.join(labels_dir, label_file)
        dst_label = os.path.join(base_dir, split, "labels", label_file)
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)

    start += count
