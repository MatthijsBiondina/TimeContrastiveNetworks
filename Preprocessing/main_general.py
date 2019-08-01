from shutil import copyfile
import os
from tqdm import tqdm
import cv2

from src.copying import Copyer
from src.gather_trials import GatherTrials
from src.segmenter import Segmenter
from src.utils.tools import Tools


IN_ROOT = ''
OU_ROOT = ''
RGB = False  # whether videos are in RGB (true) format or BGR (false) format

# GARHER TRIALS
gatherer = GatherTrials(IN_ROOT)
gatherer.walk_dirs('fileslist.txt')

# MANUAL SEGMENTATION
segmenter = Segmenter()
tasks = []
with open('fileslist.txt', 'r') as f:
    for line in f:
        tasks.append(line.replace('\n', ''))
for task in Tools.tqdm_pbar(tasks, description="SEGMENTING", total=len(tasks)):
    segmenter.segment_video(task, OU_ROOT)
os.remove('fileslist.txt')

# EXTRACT SELECTED FRAMES FROM VIDEOS
copyer = Copyer()
copyer.copy(OU_ROOT, 'left', 'middle')
copyer.copy(OU_ROOT, 'left', 'right')
copyer.copy(OU_ROOT, 'middle', 'left')

# BGR->RGB, RESIZE, AND CROP IMAGES
for trial in tqdm(Tools.list_dirs(IN_ROOT)):
    for pos in Tools.list_dirs(trial):
        for frame_pth in Tools.list_files(pos, end='.jpg'):
            if RGB:
                img = cv2.imread(frame_pth)
            else:
                img = cv2.imread(frame_pth)[:, :, ::-1]
            if (img.shape[0] == 368 and img.shape[1] == 368):
                pass
            else:
                img = cv2.resize(img, (552, 368))
                if 'left' in pos:
                    D = -398
                    img = img[:, D:D + 368, :]
                if 'middle' in pos:
                    D = 100
                    img = img[:, D:D + 368, :]
                if 'right' in pos:
                    D = 20
                    img = img[:, D:D + 368, :]
                cv2.imwrite(frame_pth, img)
