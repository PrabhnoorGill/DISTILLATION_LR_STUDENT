# RealESRGAN_Distillation/dataset.py

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
import os

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

class UnpairedLRDataset(Dataset):
    """
    PyTorch Dataset for loading unpaired Low-Resolution (LR) images.
    Used for the self-supervised feature distillation task where only LR inputs are needed.
    """
    def __init__(self, lr_dir, transform=None, max_samples=None):
        """
        Args:
            lr_dir (str): Path to the directory containing LR image files.
            transform (callable, optional): Optional transform to be applied on a sample.
                                            Defaults to converting image to tensor.
            max_samples (int, optional): If specified, limit the dataset size to this number.
        """
        self.lr_dir = lr_dir
        self.transform = transform if transform is not None else to_tensor # Use functional to avoid class dependency

        if not os.path.isdir(self.lr_dir):
            raise FileNotFoundError(f"LR directory not found: {self.lr_dir}")

        self.lr_image_files = sorted([
            os.path.join(self.lr_dir, f)
            for f in os.listdir(self.lr_dir)
            if f.lower().endswith(IMG_EXTENSIONS)
        ])

        if not self.lr_image_files:
             raise FileNotFoundError(f"No image files found in directory: {self.lr_dir}")

        # Optional: Limit dataset size for faster testing/debugging
        if max_samples is not None and max_samples < len(self.lr_image_files):
             self.lr_image_files = self.lr_image_files[:max_samples]
             print(f"Limiting dataset size to {max_samples} samples.")


    def __len__(self):
        """Returns the total number of LR images."""
        return len(self.lr_image_files)

    def __getitem__(self, idx):
        """
        Gets the LR image tensor for the given index.

        Args:
            idx (int): Index of the desired sample.

        Returns:
            torch.Tensor: LR image tensor.
                          Returns None if the image fails to load.
        """
        if idx >= len(self.lr_image_files):
             raise IndexError(f"Index {idx} out of bounds for dataset size {len(self.lr_image_files)}")

        lr_path = self.lr_image_files[idx]

        try:
            # Open image using Pillow
            lr_image = Image.open(lr_path).convert('RGB')

            # Apply transforms
            if self.transform:
                lr_tensor = self.transform(lr_image)
            else:
                # Default transform if none provided
                lr_tensor = to_tensor(lr_image)

            return lr_tensor

        except Exception as e:
            # Handle potential errors during image loading or processing
            print(f"Warning: Error processing image {lr_path}. Skipping. Error: {e}")
            # Returning None requires a custom collate_fn in the DataLoader to filter Nones,
            # or you can try returning the next valid item (less ideal for reproducibility).
            # For simplicity now, we rely on a collate_fn or checking in the training loop.
            # Alternatively, raise the exception if you want training to stop on errors.
            # Let's try returning a placeholder or skipping via collate_fn is usually preferred.
            # We will handle potential None return in the training loop's dataloader iteration.
            # return self.__getitem__((idx + 1) % len(self)) # Try next item (can cause recursion)
            return None # Simplest to handle in DataLoader or loop