import cv2
import numpy as np
import os

from src.alignment.align_matrix import AlignMatrix
from src.plot.cv_line import CVLine
from src.utils.tools import Tools


class RewardPlot:
    def __init__(self,
                 root=None):
        self.POS = tuple([Tools.fname(f)
                          for f in Tools.list_dirs(Tools.list_dirs(root)[0])])
        Tools.debug(self.POS)
        if 'steady' in self.POS:
            self.POS = 'steady'
        elif 'middle' in self.POS:
            self.POS = 'middle'
        else:
            self.POS = self.POS[0]
        am = AlignMatrix(root)
        self.alignments = am.load()

    def visualize(self, in_folder, ou_folder):
        Tools.makedirs(ou_folder)

        nn = self._nearest_neighbor(in_folder)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(os.path.join(
            ou_folder, in_folder.split('/')[-1] + '.mp4'),
            fourcc, 16, (480, 480))

        frames = [Tools.fname(f) for f in Tools.list_files(
            os.path.join(in_folder, self.POS), end='.jpg')]
        cv_plot = CVLine((240, 480), minx=0, maxx=len(frames), miny=0, maxy=1)

        main_frm = np.zeros((480, 480, 3), dtype=np.uint8)
        for fii, frame in Tools.tqdm_pbar(
                enumerate(frames), Tools.fname(in_folder), total=len(frames)):
            frame = cv2.imread(os.path.join(in_folder, self.POS, frame))
            try:
                ancho = cv2.imread(os.path.join(
                    nn, self.POS, self.alignments[in_folder][nn][fii][0]))
            except Exception as e:
                Tools.debug(self.alignments[in_folder][nn])
                Tools.debug(e, ex=0)

            frame = cv2.resize(frame, (240, 240))
            ancho = cv2.resize(ancho, (240, 240))

            main_frm[:240, :240, :] = frame[:, :, :]
            main_frm[:240, 240:, :] = ancho[:, :, :]

            R = self._reward_score(in_folder, fii)
            main_frm[240:, :, :] = cv_plot.plot((fii, R))

            writer.write(main_frm)
        writer.release()

    def _reward_score(self, trial, index):
        cum_gain = 0.
        cum_weight = 0.
        for match_trial in self.alignments[trial]:
            match_frm, weight = self.alignments[trial][match_trial][index]
            match_nmr = int(''.join(filter(str.isdigit, match_frm)))
            try:
                match_min = min(
                    int(''.join(filter(str.isdigit, Tools.fname(f))))
                    for f in Tools.list_files(
                        os.path.join(match_trial, self.POS), end='.jpg'))
                match_max = max(
                    int(''.join(filter(str.isdigit, Tools.fname(f))))
                    for f in Tools.list_files(
                        os.path.join(match_trial, self.POS), end='.jpg'))
            except Exception as e:
                Tools.debug(os.path.join(Tools.pathstr(match_trial), self.POS))
                Tools.debug(e, ex=0)

            cum_gain += (match_nmr - match_min) / (match_max - match_min)
            cum_weight += 1
        return cum_gain / cum_weight

    def _nearest_neighbor(self, trial):
        if 'fake' not in trial:  # show optimal behavior
            return min(self.alignments[trial],
                       key=lambda anchor: self._trial_distance(trial, anchor))
        else:
            # show median, because best fit does not recognize fail,
            # and worst fit has bad alignment for wrong reasons
            return sorted(self.alignments[trial],
                          key=lambda a: self._trial_distance(trial, a)
                          )[len(self.alignments[trial]) // 2]

    def _trial_distance(self, trial, anchor):
        dist = 0
        for _, weight in self.alignments[trial][anchor]:
            dist += 1 / weight
        return dist
