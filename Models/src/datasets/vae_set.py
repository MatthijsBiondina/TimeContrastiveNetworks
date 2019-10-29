import cv2
import numpy as np
import os
import random
import sys
import torch
import torch.utils.data as data

from src.datasets.transform import Transformer
from src.utils.tools import Tools
from src.utils.config import Config
poem = Tools.tqdm_pbar


class VAESet(data.Dataset):
    def __init__(self,
                 root_dir=None,
                 pos=None,
                 batch_size=Config.TCN_BATCH,
                 input_size=(3, 224, 224),
                 output_size=(3, 224, 224),
                 augment=False):
        self.root = root_dir
        assert pos is not None
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.augment = augment

        frames = []

        for trial_folder in Tools.list_dirs(root_dir):
            for frame_pth in Tools.list_files(
                    os.path.join(trial_folder, pos), end='.jpg'):
                frames.append(frame_pth)

        if augment:
            random.shuffle(frames)

        self.batches = [[]]
        for frame in poem(frames, 'LOADING ' + Tools.fname(root_dir)):
            if len(self.batches[-1]) >= batch_size:
                self.batches.append([frame])
            else:
                self.batches[-1].append(frame)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch = self.batches[idx]
        X = np.zeros((len(batch),) + self.input_size, dtype=np.float32)
        pths = []

        for ii, pth in enumerate(batch):
            img = Transformer.transform(
                cv2.imread(pth), augment=False,
                BGR=False).astype(np.float32)
            pths.append(pth)

            X[ii, :, :, :] = img.transpose(2, 0, 1) / (255 / 2) - 1.

        return torch.FloatTensor(X), torch.FloatTensor(np.copy(X)), pths
