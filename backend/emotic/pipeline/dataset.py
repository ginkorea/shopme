# backend/emotic/pipeline/dataset.py

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os

class EmoticDataset(Dataset):
    def __init__(self, csv_path, data_root, crop_type='crop', transform=None):
        """
        Args:
            csv_path (str): Path to CSV annotations.
            data_root (str): Root directory containing img_arrs/ and annot_arrs/.
            crop_type (str): Type of input image to load: 'crop' or 'context' (defaults to 'crop').
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.annotations = pd.read_csv(csv_path)
        self.data_root = data_root
        self.crop_type = crop_type
        self.transform = transform

        self.emotion_cols = [
            'Peace', 'Affection', 'Esteem', 'Anticipation', 'Engagement', 'Confidence',
            'Happiness', 'Pleasure', 'Excitement', 'Surprise', 'Sympathy', 'Doubt/Confusion',
            'Disconnection', 'Fatigue', 'Embarrassment', 'Yearning', 'Disapproval', 'Aversion',
            'Annoyance', 'Anger', 'Sensitivity', 'Sadness', 'Disquietment', 'Fear',
            'Pain', 'Suffering'
        ]
        self.vad_cols = ['Valence', 'Arousal', 'Dominance']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        # Select correct image array
        npy_filename = row['Crop_name'] if self.crop_type == 'crop' else row['Arr_name']
        img_path = os.path.join(self.data_root, 'img_arrs', npy_filename)
        image = np.load(img_path)

        # Convert HWC to CHW if needed
        if image.ndim == 3 and image.shape[-1] == 3:
            image = image.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)

        # Convert to torch tensor
        image = torch.from_numpy(image).float()

        if self.transform:
            image = self.transform(image)

        # Get multi-label emotion targets
        emotions = torch.tensor(row[self.emotion_cols].values.astype(np.float32))
        vad = torch.tensor(row[self.vad_cols].values.astype(np.float32))

        return {
            'img': image,
            'emotions': emotions,
            'vad': vad
        }

