
import os
import random
from PIL import Image

# Define the source directories
source_dir = r'C:\Users\ubuntu\Desktop\Cristina\DFD\iDFD-main\Data\Raw\inpainting'

# Define the destination directory
destination_dir = r'C:\Users\ubuntu\Desktop\Cristina\git repos\2HDED_selfsupervised\datasets\iDFD'

# Define the split ratio
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Define the crop size
crop_size = (1024, 1024)

# Ensure correspondence by listing images in all folders
all_images = {}
folders = ['All_In_Focus', 'Depth', 'Out_Of_Focus']

for folder in folders:
    folder_path = os.path.join(source_dir, folder)
    all_images[folder] = sorted([f for f in os.listdir(folder_path)])

# Check that all folders have the same number of images
num_images = len(all_images[folders[0]])
assert all(len(all_images[folder]) == num_images for folder in folders), "Folders do not contain the same number of images"

# Shuffle indices to maintain correspondence
indices = list(range(num_images))
random.shuffle(indices)

# Split indices into train, val, and test
train_split = int(train_ratio * num_images)
val_split = int(val_ratio * num_images)

train_indices = indices[:train_split]
val_indices = indices[train_split:train_split + val_split]
test_indices = indices[train_split + val_split:]

# Function to crop image
def crop_image(image_path, output_path, crop_size):
    image = Image.open(image_path)
    width, height = image.size
    left = (width - crop_size[0]) / 2
    top = (height - crop_size[1]) / 2
    right = (width + crop_size[0]) / 2
    bottom = (height + crop_size[1]) / 2
    cropped_image = image.crop((left, top, right, bottom))
    cropped_image.save(output_path)

# Process and save images
for split, split_indices in zip(['train', 'val', 'test'], [train_indices, val_indices, test_indices]):
    for folder in folders:
        split_folder_path = os.path.join(destination_dir, folder, split)
        os.makedirs(split_folder_path, exist_ok=True)

        folder_path = os.path.join(source_dir, folder)
        for index in split_indices:
            image_name = all_images[folder][index]
            image_path = os.path.join(folder_path, image_name)
            output_path = os.path.join(split_folder_path, image_name)
            crop_image(image_path, output_path, crop_size)

