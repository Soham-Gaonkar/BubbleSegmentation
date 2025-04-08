import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import re
import numpy as np
from torchvision import transforms
from config import Config
from torchvision import transforms


class UltrasoundSegmentationDataset(Dataset):
    """
    Dataset for both single-frame and sequence-based ultrasound segmentation.
    """

    def __init__(self, image_dir, label_dir, transform=None, sequence_length=1):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.sequence_length = sequence_length

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.label_files = [f for f in os.listdir(label_dir) if f.endswith('.png') or f.endswith('.jpg')]

        # Build ID to label file mapping
        self.label_dict = {
            self._extract_id(f): f for f in self.label_files
        }
        

        self.samples = []

        if self.sequence_length > 1:
            for i in range(len(self.image_files) - sequence_length + 1):
                seq_images = self.image_files[i:i + sequence_length]
                seq_ids = [self._extract_id(f) for f in seq_images]

                if all(img_id in self.label_dict for img_id in seq_ids):
                    seq_labels = [self.label_dict[img_id] for img_id in seq_ids]
                    self.samples.append((seq_images, seq_labels))
                else:
                    missing_ids = [img_id for img_id in seq_ids if img_id not in self.label_dict]
                    print(f"Skipping sequence due to missing labels for: {missing_ids}")
        else:
            for img_file in self.image_files:
                img_id = self._extract_id(img_file)
                if img_id in self.label_dict:
                    label = self.label_dict[img_id]
                    self.samples.append(([img_file], [label]))
                else:
                    print(f"Missing label for image: {img_file} (ID: {img_id})")

        print("\n=== Image–Label Mapping Debug ===")
        for i in range(min(10, len(self.samples))):
            img_files, label_files = self.samples[i]
            print(f"Sample {i+1}:")
            for img, lbl in zip(img_files, label_files):
                print(f"  Image: {img} --> Label: {lbl}")
        print("========================================\n")

    def _extract_id(self, filename):
        return os.path.splitext(filename)[0].replace("US", "").replace("Label", "")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_seq_files, lbl_filenames = self.samples[idx]

        images = []
        target_label_file = lbl_filenames[0]
        label_pil = Image.open(os.path.join(self.label_dir, target_label_file)).convert('L')
        label_transformed = None

        for i, img_file in enumerate(img_seq_files):
            image_pil = Image.open(os.path.join(self.image_dir, img_file)).convert('RGB')

            # Apply transform (to both image and label at first image only)
            if self.transform:
                if i == 0:
                    img_tensor, label_transformed = self.transform(image_pil, label_pil)
                else:
                    img_tensor, _ = self.transform(image_pil, label_pil)
            else:
                raise NotImplementedError("Transforms are required.")

            images.append(img_tensor)

        image_sequence_tensor = torch.stack(images, dim=0)

        # Always return the filename clearly along with tensors
        filename = img_seq_files[-1]  # usually, the last image in the sequence

        if self.sequence_length > 1:
            return image_sequence_tensor, label_transformed, filename
        else:
            return images[0], label_transformed, filename


class JointTransform:
    """Applies transformations to both image and label."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

class Resize:
    """Resize image and label to target size."""
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        image = image.resize(self.size, Image.BILINEAR)
        label = label.resize(self.size, Image.NEAREST)
        return image, label

# class PILToTensor:
#     """Convert PIL image and mask to torch.Tensor."""
#     def __call__(self, image, label):
#         image = transforms.ToTensor()(image)
#         label = transforms.ToTensor()(label)
#         return image, label

# PILToTensor class (only this part needs to be changed)
class PILToTensor:
    """Convert PIL image and mask to torch.Tensor and binarize label."""
    def __call__(self, image, label):
        image = transforms.ToTensor()(image)
        label = transforms.ToTensor()(label)
        label = (label > 0.5).float()  
        return image, label


class Grayscale:
    """Convert image to grayscale (not label)."""
    def __call__(self, image, label):
        image = image.convert('L')
        return image, label

def create_ultrasound_dataloaders(image_dir, label_dir, batch_size=16, val_split=0.10, num_workers=4, image_size=(1024, 256), sequence_length=1):
    """
    Create train and val DataLoaders with correct shape depending on model type.
    """


    joint_transforms = JointTransform([Resize(image_size)])
    tensor_transforms = JointTransform([Grayscale(), PILToTensor()])

    full_dataset = UltrasoundSegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=tensor_transforms,
        sequence_length=sequence_length
    )

    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    print('dataset size:',dataset_size)
    print('val_size:',val_size)
    print('train_size:',train_size)

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Set transforms
    train_dataset.dataset.transform = tensor_transforms
    val_dataset.dataset.transform = tensor_transforms

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

from config import Config

# --- Hook for train.py ---
if __name__ == "__main__":
    image_dir = Config.IMAGE_DIR
    label_dir = Config.LABEL_DIR

    train_loader, val_loader = create_ultrasound_dataloaders(
        image_dir=image_dir,
        label_dir=label_dir,
        batch_size=Config.BATCH_SIZE,
        image_size=Config.IMAGE_SIZE,
        sequence_length=Config.SEQUENCE_LENGTH
    )

    images, labels = next(iter(train_loader))
    print(f"Train - Images: {images.shape}, Labels: {labels.shape}")
    images, labels = next(iter(val_loader))
    print(f"Val - Images: {images.shape}, Labels: {labels.shape}")
    print("Dataloader ready!")
