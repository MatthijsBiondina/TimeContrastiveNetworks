import cv2
import numpy as np
import os
import tqdm as tqdm

from src.utils.tools import Tools


class Copyer:
    """
        To prevent waiting for storing frames during segmentation, we store temporaty pointer images (2x2) black image. This class extracts the actual frames from the videos. Video folders should have unique names, because duplicates are overwritten.
    """

    def __init__(self):
        self.trials = []
        with open('./res/folding_files.txt', 'r') as f:
            for line in f:
                self.trials.append(line.replace('\n', ''))

    def copy(self, root, in_pos, ou_pos):
        """
            extract frames from video files

            Args:
                root: string - path to store video files
                in_pos: string - camera perspective (e.g. left, middle, right)
                ou_pos: string - name for camera perspective directory in
                                 dataset
        """
        for trial in Tools.tqdm_pbar(Tools.list_dirs(root), 'COPYING'):
            Tools.pyout(trial)
            fname = '_'.join(Tools.fname(trial).split('_')[:-1])
            Tools.makedirs(os.path.join(trial, ou_pos))

            vid_folder = self._find(fname)
            frames = []
            for frame in Tools.list_files(os.path.join(trial, in_pos)):
                frames.append(int(Tools.fname(frame).split('.')[0]))

            path = os.path.join(
                vid_folder, 'color-recording-' + ou_pos + '.avi')
            if not os.path.isfile(path):
                path = path.replace('.avi', '-x265.mp4')
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                Tools.pyout("ERROR OPENING VideoCapture")
                raise FileNotFoundError('in copy(): "' + path + '"')

            ii = 0
            with tqdm(total=len(frames)) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret is True:
                        if ii in frames:
                            img = np.copy(frame)
                            cv2.imwrite(
                                os.path.join(trial, ou_pos,
                                             str(ii).zfill(5) + '.jpg'), img)
                            pbar.update(1)
                    else:
                        break
                    ii += 1

    def _find(self, trial_name):
        """
            find the corresponding video file given trial name

            Args:
                trial_name: string - (sub)str of video file name

            Returns:
                string - path to file
        """
        for trial in self.trials:
            if trial_name in trial:
                return trial

        raise FileNotFoundError('in find(): "' + trial_name + '"')
