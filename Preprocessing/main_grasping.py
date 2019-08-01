import cv2
import os
import shutil

from src.utils.tools import Tools
from src.segmenter_grasping import SegmenterGrasping

IN_ROOT = ''
OU_ROOT = ''
RGB = False  # whether videos are in RGB (true) format or BGR (false) format

# MANUALLY SEGMENT RECORDINGS
s = SegmenterGrasping()
for trial in Tools.tqdm_pbar(
        Tools.list_dirs(IN_ROOT), description="SEGMENTING"):
    s.segment(trial, os.path.join(OU_ROOT, Tools.fname(trial)))

# COPY ACTUAL FRAMES TO OU_ROOT
for trial in Tools.tqdm_pbar(
        Tools.list_dirs(OU_ROOT), description="MOVING FRAMES"):
    for pos in Tools.list_dirs(trial):
        with open(os.path.join(pos, 'frames.txt'), 'r') as f:
            for path in f:
                shutil.copy(
                    path.replace('\n', ''),
                    os.path.join(pos, Tools.fname(path.replace('\n', ''))))
        os.remove(os.path.join(pos, 'frames.txt'))

# BGR->RGB, RESIZE, AND CROP IMAGES
for trial in Tools.tqdm_pbar(Tools.list_dirs(IN_ROOT), description="CROPPING"):
    for pos in Tools.list_dirs(trial):
        for frame_pth in Tools.list_files(pos, end='.jpg'):
            if RGB:
                img = cv2.imread(frame_pth)
            else:
                img = cv2.imread(frame_pth)[:, :, ::-1]
            img = cv2.resize(img, (552, 368))
            if 'left' in pos:
                D = -378
                img = img[:, D:D + 368, :]
            if 'middle' in pos:
                D = 100
                img = img[:, D:D + 368, :]
            if 'right' in pos:
                D = 60
                img = img[:, D:D + 368, :]
            cv2.imwrite(frame_pth, img)
