import atexit
import curses
import cv2
import numpy as np
import os
import shutil
import signal
import sys
from tqdm import tqdm

from src.utils.tools import Tools


class SegmenterGrasping:
    """
        Class for scrolling through video files in grasping data manually and
        defining starting and end points of segments.
        Slightly modified from general tool, in that this tool allows
        extraction of two datasets simultaneously, grasping and grasping
        (reverse). This is perfectly possible with the general tool, but it
        would require two separate passes through the data for each task.

        Controls:
        One can move through the video using the d, e, 3, a, q, and 1 keys;
        define an anchor point with the u key; and save all frames between the
        anchor and current frames as a trial with the [ and ] keys.

        +   go to next video in data folder

        > scrolling:
        d   move forward one frame
        e   move forward five frames
        3   move forward twenty-five frames
        a   move backward one frame
        q   move backward five frames
        1   move backward twenty-five frames

        > segmentation:
        u   define new anchor point - frames between previous and new anchor
            point are not stored in any dataset.
        ]   save currently selected trial in 'forwards' dataset and define
            current frame as new anchor point
        [   save currently selected trial in 'backwards' dataset and define
            current frame as new anchor point
        .   undo - undo previous action and set anchor point to previous
            anchor point
    """

    def __init__(self, npos=3):
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(1)
        self.stdscr.refresh()
        self._hangle_exit()

        self.main_frm = np.zeros((240, 360, 3), dtype=np.uint8)
        self.zfill_n = None

    def segment(self, in_root, ou_root):
        """
            Extract trials from video

            Args:
                in_root: string - path to input trial folder
                ou_root: string - path to output folder root
        """
        # list containing history of anchor points
        progression = [0]
        folders = []
        try:
            frames, N = self._load_images(in_root)
            n = 0
            key = ''
            feedback = ''
            with tqdm(total=N) as pbar:
                while key != ord('+'):
                    self.main_frm[:, :, :] = frames[n, :, :, ::-1]
                    if feedback == 'fwd':
                        # if saved segment in forward dataset:
                        #    display green circle for feedback
                        self._feedback((0, 255, 0))
                    if feedback == 'bwd':
                        # if saved segment in backward dataset:
                        #    display magenta circle for feedback
                        self._feedback((255, 0, 220))
                    if feedback == 'skip':
                        # if new anchor point set without save:
                        #    display blue circle for feedback
                        self._feedback((255, 0, 0))
                    if feedback == 'undo':
                        # if undo:
                        #    display red circle for feedback
                        self._feedback((0, 0, 255))

                    # render and await user input
                    Tools.render(self.main_frm)
                    feedback = ''
                    key = self.stdscr.getch()
                    self.stdscr.addch(20, 25, key)
                    self.stdscr.refresh()

                    # process user input
                    n = self._process_nav(key, n, N, progression)
                    save = self._process_save(key, in_root, ou_root, folders,
                                              progression[-1], n, progression)
                    if save:
                        feedback = save
                    if self._process_skip(key, folders, n, progression):
                        feedback = 'skip'
                    n, undo = self._process_undo(key, progression, folders, n)
                    if undo:
                        feedback = 'undo'
                    pbar.update(n - pbar.n)

        except Exception as e:
            self._shutdown('Shutting down due to exception')
            raise e

    def _process_nav(self, key, n, N, prog):
        """
            process user input related to scrolling

            Args:
                key:  int  - unicode number of pressed key
                n:    int  - current position
                N:    int  - position of last frame in video
                prog: list - anchor position history

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
            n = max(prog[-1], n - 1)
        elif key == ord('q'):  # bwd x5
            n = max(prog[-1], n - 5)
        elif key == ord('1'):  # bwd x25
            n = max(prog[-1], n - 25)
        return n

    def _process_undo(self, key, progression, folders, n):
        """
            process user input related to undo

            Args:
                key:         int  - unicode number of pressed key
                progression: list - anchor position history
                folders:     list - history of created output folders
                n:           int  - current position

            Returns:
                n: int  - new position
                >  bool - whether undo action has been performed
        """
        if key == ord('.'):
            if len(progression) > 1:
                if folders[-1] is not None:
                    shutil.rmtree(folders[-1])
                progression.pop(-1)
                return progression[-1], True
        return n, False

    def _store(self, in_root, path, frm, to):
        """
            Store selected fragment. Creates file containing paths to frames.

            Args:
                in_root: string - path to input folder
                path:    string - path to output folder
                frm:     int    - first frame of trial
                to:      int    - last frame of trial
        """
        Tools.makedirs(path)

        for pos in ('left', 'middle', 'right'):
            Tools.makedirs(os.path.join(path, pos))
            with open(os.path.join(path, pos, 'frames.txt'), 'w+') as f:
                for ii in range(frm, to + 1):
                    f.write(os.path.join(
                        in_root, pos, str(ii).zfill(self.zfill_n) + '.jpg') +
                        '\n')

    def _load_images(self, path):
        """
            read images into working memoryk

            Args:
                path: string - input path

            Returns:
                imgs: list(np.array) - list of frames from 'left' perspective
                N:    int            - number of frames
        """
        N = len(Tools.list_files(os.path.join(path, 'left')))
        imgs = np.zeros((N, 240, 360, 3), dtype=np.uint8)

        load_path = os.path.join(path, 'left')
        for fii, frm_pth in Tools.tqdm_pbar(
                enumerate(Tools.list_files(load_path)),
                path, total=N):
            if self.zfill_n is None:
                self.zfill_n = len(Tools.fname(frm_pth).split('.')[0])

            img = cv2.imread(frm_pth)
            img = cv2.resize(img, (360, 240))
            imgs[fii, :, :, :] = np.copy(img[:, :, :])
        return imgs, N

    def _feedback(self, color):
        """
            draw circle on displayed frame for feedback on action
        """
        cv2.circle(self.main_frm, (10, 10), 10, color, -1)

    def _hangle_exit(self):
        """
            refer to shutdown script in case of exit
        """
        atexit.register(self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

    def _write(self, string):
        """
            write string compatibly with curses
        """
        self.stdscr.addstr(10, 0, string)

    def _shutdown(self, *args):
        """
            do nice shutdown of curses to restore terminal to normal behavior
        """
        self.stdscr.move(0, 0)
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.endwin()

    def _debug(self, message):
        """
            shutdown curses, print message, then exit
        """
        self._shutdown()
        Tools.pyout(message)
        sys.exit(0)
