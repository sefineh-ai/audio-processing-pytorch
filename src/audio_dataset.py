import os
import numpy as np
import torch
from torch.utils.data import Dataset

class AudioFeatureDataset(Dataset):
    def __init__(self, features_root):
        self.samples = []
        self.class_to_idx = {}
        for idx, class_name in enumerate(sorted(os.listdir(features_root))):
            class_dir = os.path.join(features_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            self.class_to_idx[class_name] = idx
            for fname in os.listdir(class_dir):
                if fname.endswith('.npy'):
                    self.samples.append((os.path.join(class_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_path, label = self.samples[idx]
        features = np.load(feature_path)
        features = torch.tensor(features, dtype=torch.float32)
        return features, label 