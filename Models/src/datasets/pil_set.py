import cv2
import numpy as np
from PIL import Image
import random
import torch
import torch.utils.data as data
from torchvision import transforms

from src.datasets.transform import Transformer
from src.utils.config import Config
from src.utils.tools import Tools


class PILSet(data.Dataset):
    pos2num = {
        'left': 0, 'middle': 1, 'right': 2, 'bax_left': 3,
        'bax_middle': 4, 'bax_right': 5, 'steady': 6, 'mobile': 7}

    def __init__(self,
                 root_dir='./res/datasets/folding/train',
                 batch_size=Config.TCN_BATCH,
                 input_size=(3,) + Config.TCN_IMG_SIZE,
                 pos_range=Config.TCN_POS_RANGE,
                 negative_multiplier=Config.TCN_NEGATIVE_MULTIPLIER,
                 transform=None,  # unused
                 augment=False):

        self.root = root_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.pos_range = pos_range
        self.m = negative_multiplier
        self.augment = augment
        self.trial_names = []
        self.seeds = []  # use same seed when sampling eval
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.255])])

        for f in Tools.list_dirs(root_dir):
            if augment:
                self.trial_names.append(f)
            else:
                self.trial_names.append(f)
                self.seeds.append(random.randint(0, 9001))

    def __len__(self):
        return len(self.trial_names)

    def __getitem__(self, idx):
        # if val set, use same random seed
        if not self.augment:
            random.seed(self.seeds[idx])
        trial_folder = self.trial_names[idx]
        if 'fake' in trial_folder:
            return (None, None, None, trial_folder)

        X = np.zeros((self.batch_size,) + self.input_size)
        labels = np.zeros((self.batch_size))
        perspectives = np.zeros((self.batch_size,))
        paths = []
        frames_used = [-float("inf")]
        n = 0
        fails = 0
        while n < self.batch_size // 2:
            # sample two perspectives
            samples_pos = random.sample(Tools.list_dirs(trial_folder), 2)

            # sample anchor frame
            a_val, a_pth, a_idx = self._sample_frame(
                samples_pos[0], frames_used)
            # sample positive frame
            p_val, p_pth, p_idx = self._sample_frame(
                samples_pos[1], frames_used, anchor_idx=a_idx)

            # deal with failing to find a valid pair
            if not a_val or not p_val:
                fails += 1
                if fails > self.batch_size:  # give up
                    break
            else:
                # add anchor frame to batch
                paths.append(a_pth)
                img_a = Transformer.transform(
                    cv2.imread(a_pth), BGR=False)
                X[n * 2, :, :,
                    :] = self.transform(Image.fromarray(img_a)).numpy()
                labels[n * 2] = n
                perspectives[n * 2] = self.pos2num[a_pth.split('/')[-2]]

                # add positive frame to batch
                paths.append(p_pth)
                img_p = Transformer.transform(
                    cv2.imread(p_pth), BGR=False)
                X[n * 2 + 1, :, :,
                    :] = self.transform(Image.fromarray(img_p)).numpy()
                labels[n * 2 + 1] = n
                perspectives[n * 2 + 1] = self.pos2num[p_pth.split('/')[-2]]

                n += 1

        # if batch is not entirely full, cut off zero padding
        X = X[:n * 2, :, :, :]
        labels = labels[:n * 2]
        perspectives = perspectives[:n * 2]
        if X.shape[0] == 0:
            return (None, None, None, trial_folder)
        else:
            X = torch.FloatTensor(X)
            labels = torch.FloatTensor(labels)
            perspectives = torch.FloatTensor(perspectives)

            assert not Tools.contains_nan(X)
            assert not Tools.contains_nan(labels)
            assert not Tools.contains_nan(perspectives)

            return (X, labels, perspectives, paths)

    def _sample_frame(self, pos_folder, used_frames, anchor_idx=None):
        valid = False
        # frame_pth = None
        if anchor_idx is None:
            all_frames = Tools.list_files(pos_folder, end='.jpg')
        else:
            # check that current frame is a TP to anchor frame
            all_frames = [
                file for file in Tools.list_files(pos_folder, end='.jpg') if
                abs(int(file.split('/')[-1].split('.')[0]) - anchor_idx) <
                self.pos_range]
        random.shuffle(all_frames)
        for frame_pth in all_frames:
            # check that current frame is not a FP to any used frame
            idx = int(frame_pth.split('/')[-1].split('.')[0])

            frame_pth = random.choice(all_frames)
            if all([abs(idx - u_idx) > self.m * self.pos_range for
                    u_idx in used_frames]):
                valid = True
                break

        if valid:
            frame_idx = int(frame_pth.split('/')[-1].split('.')[0])
            return True, frame_pth, frame_idx
        else:
            return False, None, None

    def close(self):
        pass
