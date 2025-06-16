import os
import json
import shutil
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

# ===============================================================
# GLOBAL CONFIGURATION VARIABLES
# ===============================================================
# --- Set these paths before running the script ---

# The root directory containing 'annotations.json'
ROOT_DIR = './complete_dataset'

USE_2K_DATASET = True

# The directory where all source images are stored
IMAGES_DIR = './complete_dataset/chessred2k' if USE_2K_DATASET else './complete_dataset/chessred'

# The parent directory where the new partitioned folders will be created
OUTPUT_DIR = f'./complete_dataset/partitioned_chess{"2k" if USE_2K_DATASET else ""}_images'

# The specific partition to extract ('train', 'valid', or 'test')
PARTITION_TO_EXTRACT = 'test'

# ===============================================================
# DATASET CLASS (Simplified version)
# ===============================================================
class ChessDataset(Dataset):
    """
    Dataset class to load and partition chess images based on annotations.
    This version is simplified to only handle file names for partitioning.
    """
    def __init__(self, root_dir, images_dir, partition, use_2k_dataset=False):
        annotations_path = os.path.join(root_dir, 'annotations.json')
        print(f"Loading annotations from: {annotations_path}")
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations file not found at {annotations_path}")
        
        with open(annotations_path) as f:
            self.anns = json.load(f)

        self.root = root_dir
        self.images_dir = images_dir
        self.ids = []
        self.file_names = []
        
        for x in self.anns['images']:
            # The 'path' key might contain subdirectories, so we use it directly
            self.file_names.append(x['path'])
            self.ids.append(x['id'])
            
        self.file_names = np.asarray(self.file_names)
        self.ids = np.asarray(self.ids)

        splits = self.anns["splits"]["chessred2k"] if use_2k_dataset else self.anns["splits"]
        if partition == 'train':
            self.split_ids = np.asarray(splits['train']['image_ids']).astype(int)
        elif partition == 'valid':
            self.split_ids = np.asarray(splits['val']['image_ids']).astype(int)
        else:
            self.split_ids = np.asarray(splits['test']['image_ids']).astype(int)

        intersect = np.isin(self.ids, self.split_ids)
        self.split_indices = np.where(intersect)[0]

        # Filter the file names to match the partition
        self.file_names = self.file_names[self.split_indices]
        
        print(f"Initialized dataset for '{partition}' partition.")
        print(f"Number of images in this partition: {len(self.file_names)}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, i):
        # This method is not used for copying files but is required by the Dataset class
        image_path = os.path.join(self.images_dir, self.file_names[i])
        image = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_4)
        
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        return image

# ===============================================================
# MAIN SCRIPT LOGIC
# ===============================================================
def create_partition_folder():
    """
    Initializes the dataset for the specified partition and copies
    the corresponding images to a new directory.
    """
    print("--- Starting Image Partition Script ---")
    
    # 1. Create the dataset for the specified partition
    try:
        partition_dataset = ChessDataset(
            root_dir=ROOT_DIR,
            images_dir=IMAGES_DIR,
            partition=PARTITION_TO_EXTRACT,
            use_2k_dataset=USE_2K_DATASET
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your ROOT_DIR and IMAGES_DIR are set correctly.")
        return
    except KeyError as e:
        print(f"Error: Missing key {e} in annotations.json. The file might be structured differently.")
        return

    # 2. Create the destination directory
    destination_folder = os.path.join(OUTPUT_DIR, PARTITION_TO_EXTRACT)
    os.makedirs(destination_folder, exist_ok=True)
    print(f"Output directory created/ensured: {destination_folder}")

    # 3. Copy files
    copied_count = 0
    skipped_count = 0
    print(f"Starting to copy {len(partition_dataset.file_names)} files...")

    for file_path in partition_dataset.file_names:
        # The file_path from annotations might contain subdirectories.
        # os.path.join handles this correctly.
        source_file = os.path.join(IMAGES_DIR, file_path)
        
        # To avoid creating subdirectories in the destination, we'll
        # just use the base name of the file for the destination.
        destination_file = os.path.join(destination_folder, os.path.basename(file_path))

        if os.path.exists(source_file):
            shutil.copy(source_file, destination_file)
            copied_count += 1
        else:
            print(f"Warning: Source file not found, skipping: {source_file}")
            skipped_count += 1

    print("\n--- Script Finished ---")
    print(f"Successfully copied {copied_count} images.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} files (not found).")
    print(f"All '{PARTITION_TO_EXTRACT}' partition images are now in: {destination_folder}")


if __name__ == '__main__':
    create_partition_folder()
