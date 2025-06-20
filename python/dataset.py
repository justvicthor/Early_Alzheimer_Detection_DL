import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import pandas as pd
import scipy.ndimage

class ADNIDataset(Dataset):
    def __init__(self,
                 scans_dir,
                 tsv_path,
                 mode='train',
                 num_classes=3,
                 use_augmentation=True,
                 crop_size=96,
                 blur_sigma_range=(0.0, 1.5)):
        """
        Args:
            scans_dir (string): Directory with all the MRI scans
            tsv_path (string): Path to the tsv file with labels
            mode (string): 'train' or 'val'
            num_classes (int): Number of classes (2 or 3)
        """
        self.scans_dir = scans_dir
        self.mode = mode
        self.num_classes = num_classes

        self.df = pd.read_csv(tsv_path, sep='\t')

        if num_classes == 3:
            valid_labels = ["CN", "LMCI", "AD"]
        else:
            valid_labels = ["CN", "AD"]

        self.df = self.df[self.df['diagnosis'].isin(valid_labels)]
        self.label_map = {label: idx for idx, label in enumerate(valid_labels)}

        self.use_augmentation = use_augmentation
        self.crop_size = crop_size
        self.blur_sigma_range = blur_sigma_range

    def __len__(self):
        return len(self.df)

    def randomCrop(self, image, crop_d, crop_h, crop_w):
        d, h, w = image.shape
        if d < crop_d or h < crop_h or w < crop_w:
            raise ValueError("Crop size is larger than image size.")
        d1 = np.random.randint(0, d - crop_d + 1)
        h1 = np.random.randint(0, h - crop_h + 1)
        w1 = np.random.randint(0, w - crop_w + 1)
        return image[d1:d1+crop_d, h1:h1+crop_h, w1:w1+crop_w]

    def centerCrop(self, image, crop_d, crop_h, crop_w):
        d, h, w = image.shape
        d1 = (d - crop_d) // 2
        h1 = (h - crop_h) // 2
        w1 = (w - crop_w) // 2
        return image[d1:d1+crop_d, h1:h1+crop_h, w1:w1+crop_w]

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            scan_path = os.path.join(self.scans_dir,
                                   "subjects",
                                   row['participant_id'],
                                   row['session_id'],
                                   't1/spm/segmentation/normalized_space')

            if not os.path.exists(scan_path):
                raise FileNotFoundError(f"Missing scan path for idx {idx}: {scan_path}")

            scan_files = [f for f in os.listdir(scan_path) if 'Space_T1w' in f]

            if not scan_files:
                raise FileNotFoundError(f"No scan file in {scan_path}")

            scan_file = scan_files[0]
            scan_path_full = os.path.join(scan_path, scan_file)

            try:
                image = nib.load(scan_path_full).get_fdata()

            except Exception as e:
                raise RuntimeError(f"Error loading scan {scan_path_full}: {e}")
            image = np.nan_to_num(image)
            image = (image - image.min()) / (image.max() - image.min() + 1e-6)

            # ----------------------- AUGMENTATION -----------------------
            if self.mode == 'train' and self.use_augmentation:
                if np.random.rand() < 0.5:
                    sigma = np.random.uniform(*self.blur_sigma_range)
                    image = scipy.ndimage.gaussian_filter(image, sigma=sigma)
                # random crop sempre, blur solo col 50 %
                image = self.randomCrop(image,
                                        self.crop_size,
                                        self.crop_size,
                                        self.crop_size)
            else:
                image = self.centerCrop(image,
                                        self.crop_size,
                                        self.crop_size,
                                        self.crop_size)
            # ------------------------------------------------------------

            image = torch.FloatTensor(image).unsqueeze(0)

            label = self.label_map[row['diagnosis']]


            return image, label

        except Exception as e:
            print(f"[ERROR] Exception in __getitem__ for idx {idx}: {str(e)}")
            raise