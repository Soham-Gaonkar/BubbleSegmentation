import os
import shutil
import random
from tqdm import tqdm

# Set the source directories
train_images_dir = 'US_2'
train_labels_dir = 'Labels_2'
test_images_dir = 'US_Test_2023April7'
test_labels_dir = 'Labels_Test_2023April7'

# Set the destination directories
new_train_images_dir = 'train_images'
new_train_labels_dir = 'train_labels'
new_test_images_dir = 'test_images'
new_test_labels_dir = 'test_labels'

# Create destination directories if they don't exist
for d in [new_train_images_dir, new_train_labels_dir, new_test_images_dir, new_test_labels_dir]:
    os.makedirs(d, exist_ok=True)

# Function to get list of files from a folder (ignoring hidden files)
def get_files(directory, extension):
    return [f for f in os.listdir(directory) if f.endswith(extension) and not f.startswith('.')]

# Function to generate corresponding label filename from image filename
def get_label_filename(img_filename):
    # Replace 'US' with 'Label' and change extension to .png
    return img_filename.replace('US', 'Label').rsplit('.', 1)[0] + '.png'

# Gather all image files from both training and test directories
all_image_label_pairs = []

for img_file in get_files(train_images_dir, '.jpg'):
    label_file = get_label_filename(img_file)
    img_path = os.path.join(train_images_dir, img_file)
    label_path = os.path.join(train_labels_dir, label_file)
    all_image_label_pairs.append((img_path, label_path))

for img_file in get_files(test_images_dir, '.jpg'):
    label_file = get_label_filename(img_file)
    img_path = os.path.join(test_images_dir, img_file)
    label_path = os.path.join(test_labels_dir, label_file)
    all_image_label_pairs.append((img_path, label_path))

# Validate existence of label files
valid_pairs = []
for img_path, label_path in all_image_label_pairs:
    if os.path.exists(label_path):
        valid_pairs.append((img_path, label_path))
    else:
        print(f"Warning: Label file not found for image {img_path}")

# Shuffle the valid image-label pairs
random.shuffle(valid_pairs)

# Split into training and testing sets
split_ratio = 0.6
split_index = int(len(valid_pairs) * split_ratio)
train_pairs = valid_pairs[:split_index]
test_pairs = valid_pairs[split_index:]

print(f"Total valid pairs: {len(valid_pairs)}")
print(f"Training pairs: {len(train_pairs)}")
print(f"Testing pairs: {len(test_pairs)}")

# Copy image-label pairs to respective folders with progress bar
def copy_pairs_with_progress(pairs, image_dest_dir, label_dest_dir):
    for img_path, label_path in tqdm(pairs, desc="Copying files", unit="pair"):
        shutil.copy(img_path, os.path.join(image_dest_dir, os.path.basename(img_path)))
        shutil.copy(label_path, os.path.join(label_dest_dir, os.path.basename(label_path)))

copy_pairs_with_progress(train_pairs, new_train_images_dir, new_train_labels_dir)
copy_pairs_with_progress(test_pairs, new_test_images_dir, new_test_labels_dir)

print("Data split and copy completed.")