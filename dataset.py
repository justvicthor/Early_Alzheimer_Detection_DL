import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import pandas as pd

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
            valid_labels = ["CN", "MCI", "AD"]
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
                                   row['participant_id'],
                                   row['session_id'],
                                   't1/spm/segmentation/normalized_space')
            
            scan_file = [f for f in os.listdir(scan_path) if 'Space_T1w' in f][0]
            scan_path = os.path.join(scan_path, scan_file)
            
            image = nib.load(scan_path).get_fdata()
            image = np.nan_to_num(image)
            image = (image - image.min()) / (image.max() - image.min() + 1e-6)
            
            image = torch.FloatTensor(image).unsqueeze(0)

            label = self.label_map[row['diagnosis']]
            
            return image, label
            
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            return None 