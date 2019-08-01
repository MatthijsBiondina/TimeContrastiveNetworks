import atexit
import curses
import cv2
from copy import deepcopy
import json
import numpy as np
import os
import signal
from tqdm import tqdm

from src.plot.cv_hist import CVHist
from src.utils.tools import Tools


class Labeler:
    """
        Manually label frames in videos
    """

    def __init__(self, npos=3, labels=('isolated_grasping',
                                       'unfold',
                                       'flatten',
                                       'folding_progress',
                                       'stack')):
        self.zfill_n = None
        self.labels = labels
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        self.stdscr.refresh()
        self._hangle_exit()

        try:
            self.main_frm = np.zeros((480, 240 * npos, 3),
                                     dtype=np.uint8)
            self.cv_hist = CVHist(shape=(240, 240 * npos),
                                  labels=labels,
                                  minvls=tuple(0 for _ in labels),
                                  maxvls=(1, 1, 1, 10, 1),
                                  n_bars=1)
        except Exception as e:
            self._shutdown('foo')
            raise e

    def label_video(self, in_folder):
        """
            Have user label frames in video

            Args:
                in_folder: string - path to root of trial
        """
        if os.path.isfile(in_folder + '.json'):
            return
        try:
            frames, N = self._load_images(in_folder)

            # init label values at 0
            lvals = [(0,) * len(self.labels) for _ in range(N)]
            n = 0
            c = True
            self._render(frames, n, c, lvals[n])

            key = ''
            with tqdm(total=N) as pbar:
                while key != ord('+'):
                    n_ = n
                    key = self.stdscr.getch()
                    self.stdscr.addch(20, 25, key)
                    self.stdscr.refresh()
                    n = self._process_nav(key, n, N)
                    c = self._process_cop(key, c)
                    if c:
                        for nii in range(min(n_, n), max(n_, n) + 1):
                            lvals[nii] = deepcopy(lvals[n_])
                    lvals[n] = self._process_vals(key, lvals[n])

                    pbar.update(n - pbar.n)
                    self._render(frames, n, c, lvals[n])
            self._save_labels(in_folder, lvals)
        except Exception as e:
            self._shutdown('foo')
            raise e

    def _process_nav(self, key, n, N):
        """
            process user input related to navigation

            Args:
                key: int - unicode number of keypress
                n:   int - current position
                N:   int - nr of frames in buffer

            Returns:
                n: int - new position
        """
        if key == ord('d'):  # fwd
            n = min(n + 1, N - 1)
        elif key == ord('e'):  # fwd x5
            n = min(n + 5, N - 1)
        elif key == ord('3'):  # fwd x25
            n = min(n + 25, N - 1)
        elif key == ord('a'):  # bwd
            n = max(0, n - 1)
        elif key == ord('q'):  # bwd x5
            n = max(0, n - 5)
        elif key == ord('1'):  # bwd x25
            n = max(0, n - 25)
        return n

    def _process_cop(self, key, c):
        """
            process user input related to copy mode

            Args:
                key: int  - unicode number of keypress
                c:   bool - whether copy mode is on

            Returns:
                bool - new state of copy mode
        """
        if key == ord('x'):  # switch copy mode
            c = not c
        return c

    def _process_vals(self, key, lvals):
        """
            process user input related to label values

            Controls:
            f/r -> move 1st label value up/down
            g/t -> move 2nd label value u/down
            ...
            etc.

            Args:
                key: int    - unicode number of keypress
                lvals: list - current values

            Returns
                list - new values
        """
        plus = ('r', 't', 'y', 'u', 'i', 'o', 'p')
        min_ = ('f', 'g', 'h', 'j', 'k', 'l', ';')
        new_vals = list(lvals)
        for lii in range(len(lvals)):
            if key == ord(plus[lii]):
                new_vals[lii] += 1
            elif key == ord(min_[lii]):
                new_vals[lii] -= 1
        return tuple(new_vals)

    def _render(self, frames, n, c, lvals):
        """
            render current frames with histograms of label values

            Args:
                frames: list - containing frames
                n:      int  - current position
                c:      bool - state of copy mode
                lvals:  list - current label values
        """
        try:
            for pii, pos in enumerate(frames):
                self.main_frm[:240, pii * 240:(pii + 1) * 240, :] = \
                    frames[pos][n, :, :, ::-1]

            self.main_frm[240:, :, :] = self.cv_hist.plot(lvals)

            if c:
                cv2.circle(self.main_frm, (10, 10), 10, (0, 255, 0), -1)
            else:
                cv2.circle(self.main_frm, (10, 10), 10, (0, 0, 255), -1)

            Tools.render(self.main_frm)
        except Exception as e:
            self._shutdown()
            raise e

    def _save_labels(self, path, lvals):
        """
            store label allocations allong with trial

            Args:
                path:  string - path to trial folder
                lvals: list   - values of labels in each frame
        """
        max_ = tuple(
            [lvals[max(range(len(lvals)), key=lambda x: lvals[x][lii])][lii]
             for lii in range(len(lvals[0]))])
        lbl_dict = {}
        for lii, lval in enumerate(lvals):
            key = str(lii).zfill(self.zfill_n) + '.jpg'
            lbl_dict[key] = {}
            for vii, val in enumerate(lval):
                lbl_dict[key][self.labels[vii]] = val / max(1, max_[vii])
        with open(path + '.json', 'w+') as f:
            json.dump(lbl_dict, f, indent=1)

    def _load_images(self, path):
        """
            load images in working memory

            Args:
                path: string - path to trial folder

            Returns:
                imgs: list - cv2 images
                N:    int  - number of frames
        """
        imgs = {}
        N = len(Tools.list_files(Tools.list_dirs(path)[0]))
        for pos in Tools.tqdm_pbar(Tools.list_dirs(path), Tools.fname(path)):
            imgs[Tools.fname(pos)] = np.zeros((N, 240, 240, 3), dtype=np.uint8)
            for fii, frm_pth in enumerate(Tools.list_files(pos)):
                if self.zfill_n is None:
                    self.zfill_n = len(Tools.fname(frm_pth).split('.')[0])
                img = cv2.imread(frm_pth)
                img = cv2.resize(img, (240, 240))
                imgs[Tools.fname(pos)][fii, :, :, :] = np.copy(img[:, :, :])
        return imgs, N

    def _hangle_exit(self):
        """
            refer to shutdown in case of exit
        """
        atexit.register(self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

    def _err(self, e):
        self._shutdown
        raise e

    def _shutdown(self, *args):
        """
            properly shutdown curses to restore normal terminal behavior
        """
        self.stdscr.move(0, 0)
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.endwin()
