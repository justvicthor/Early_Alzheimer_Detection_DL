import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import pandas as pd
import scipy.ndimage

class ADNIDataset(Dataset):
    def __init__(self, scans_dir, tsv_path, mode='train', num_classes=3):
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

    def __len__(self):
        return len(self.df)

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

            # Random Gaussian Blurring (Data Augmentation)
            if self.mode == 'train':
                if np.random.rand() < 0.5: 
                    sigma = np.random.uniform(0, 1.5)
                    image = scipy.ndimage.gaussian_filter(image, sigma=sigma)

            image = torch.FloatTensor(image).unsqueeze(0)

            label = self.label_map[row['diagnosis']]


            return image, label

        except Exception as e:
            print(f"[ERROR] Exception in __getitem__ for idx {idx}: {str(e)}")
            raise