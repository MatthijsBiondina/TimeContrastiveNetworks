import cv2
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms

from src.datasets.transform import Transformer
from src.utils.config import Config
from src.utils.tools import Tools


class EmbedderSet(data.Dataset):
    def __init__(self,
                 root_dir=None,
                 batch_size=Config.TCN_BATCH,
                 input_size=(3,) + Config.TCN_IMG_SIZE):
        self.root = root_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.255])])
        self.frame_names = Tools.list_files(self.root)
        self.batch_paths = [[]]
        for frame_name in sorted(Tools.list_files(self.root, end='.jpg')):
            if len(self.batch_paths[-1]) >= self.batch_size:
                # start new batch
                self.batch_paths.append([frame_name])
            else:
                # append to last batch
                self.batch_paths[-1].append(frame_name)

    def __len__(self):
        if self.batch_paths == [[]]:
            return 0
        else:
            return len(self.batch_paths)

    def __getitem__(self, idx):
        paths = self.batch_paths[idx]
        X = np.zeros((len(paths),) + self.input_size, dtype=np.float32)
        for ii, path in enumerate(paths):
            img = cv2.imread(path)
            if np.isnan(img).any():
                Tools.pyout('nan in', path)
            img = Transformer.transform(cv2.imread(
                path), augment=False, BGR=False)
            X[ii, :, :, :] = self.transform(Image.fromarray(img)).numpy()

        if np.isnan(X).any():
            Tools.pyout('nan in tensor')
        return torch.FloatTensor(X), paths

    def close(self):
        Tools.log("Closing EmbedderSet")
