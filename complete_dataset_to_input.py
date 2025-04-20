import json
import os

image_files = []
base_path = os.path.join("complete_dataset", "chessred2k", "images")

for dir in sorted(os.listdir(base_path)):
    dir_path = os.path.join(base_path, dir)
    for image_name in sorted(os.listdir(dir_path)):
        image_path = os.path.join(dir_path, image_name)
        image_files.append(image_path)

with open("input.json", "w") as f:
    json.dump({
        "image_files": image_files
    }, f, indent=4)
