"""
Author : Tanmay Jain
"""

import torch  
from PIL import Image 
from torch.utils.data import Dataset


class DatasetImages(Dataset):
    """
    Dataset loading images on hard drive

    Parameters
    ----------
    path : pathlib.Path
        Path to folder containing all the images

    transform : None or callabel 
        The transform to be applied to the images(Conversion to tensor and Resizing )
    
    Attributes
    ----------
    all_paths : List 
        List of all paths to '.jpg' images
    """
    def __init__(self , path , transform):
        super().__init__()
        self.all_paths = sorted([p for p in path.iterdir() if p.suffix == ".jpg"])
        self.transform = transform

    def __getitem__(self, index):
        """
        Get a single item
        """
        img = Image.open(self.all_paths[index])
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        """
        Compute the length of the dataset
        """
        return len(self.all_paths)
