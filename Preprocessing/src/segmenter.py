import atexit
import curses
import cv2
import numpy as np
import os
import shutil
import signal
import time
from tqdm import tqdm

from src.utils.tools import Tools


class Segmenter:
    """
        Class for scrolling through video files and defining starting and end points of segments.

        Controls:
        One can move through the video using the d, e, 3, a, q, and 1 keys; define an anchor point with the u key; and save all frames between the anchor and current frame as a trial with the ] key.

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
            point are not stored in the dataset
        ]   save currently selected trial in output folder
        .   undo - undo previous action and set anchor point to previous anchor
            point
    """

    def __init__(self):
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(1)
        self.stdscr.refresh()
        self._hangle_exit()

    def segment_video(self, in_folder, ou_folder):
        """
            Extract trials from video

            Args:
                in_folder: string - path to intput video folder
                ou_fodler: string - path to output folder root
        """
        try:
            l_color = self._load_videos(in_folder)[0]

            img = np.copy(l_color[0])
            img = cv2.resize(img, (240, 160))
            Tools.render(img[:, :, ::-1])

        except Exception as e:
            print(in_folder)
            print(e)
            time.sleep(1000)
            raise e

        progression = [0]
        created_folders = []
        ii = 0
        key = ''
        try:
            with tqdm(total=len(l_color)) as pbar:
                while key != ord('+'):
                    feedback = ''
                    key = self.stdscr.getch()
                    self.stdscr.addch(20, 25, key)
                    self.stdscr.refresh()
                    if key == ord('d'):
                        # move forwards slow
                        ii = min(ii + 1, len(l_color) - 1)
                    elif key == ord('e'):
                        # move forwards medium
                        ii = min(ii + 5, len(l_color) - 1)
                    elif key == ord('3'):
                        # move forwards fast
                        ii = min(ii + 25, len(l_color) - 1)

                    elif key == ord('a'):
                        # move backwards slow
                        ii = max(progression[-1], ii - 1)
                    elif key == ord('q'):
                        # move backwards medium
                        ii = max(progression[-1], ii - 5)
                    elif key == ord('1'):
                        # move backwards fast
                        ii = max(progression[-1], ii - 25)

                    elif key == ord(']'):
                        # STORE SEGMENT
                        # make new folder
                        store_folder_number = len([
                            f for f in created_folders if f is not None])
                        savepath = os.path.join(ou_folder, Tools.fname(
                            in_folder) + '_' + str(store_folder_number).zfill(3))
                        self._makedirs(savepath)

                        # store images
                        self._store_imgs(l_color, savepath +
                                         '/left', progression[-1], ii)

                        # update progression data
                        progression.append(ii)
                        created_folders.append(savepath)

                        feedback = 'store'

                    elif key == ord('u'):
                        # skip
                        progression.append(ii)
                        created_folders.append(None)

                        feedback = 'skip'
                    elif key == ord('.'):
                        # undo
                        if len(progression) > 1:
                            if created_folders[-1] is not None:
                                shutil.rmtree(created_folders[-1])

                            progression.pop(-1)
                            created_folders.pop(-1)
                            ii = progression[-1]

                        feedback = 'back'

                    pbar.update(ii - pbar.n)
                    img = np.copy(l_color[ii])
                    img = cv2.resize(img, (240, 160))

                    if feedback == 'store':
                        cv2.circle(img, (10, 10), 10, (0, 255, 0), -1)
                    if feedback == 'skip':
                        cv2.circle(img, (10, 10), 10, (255, 0, 0), -1)
                    if feedback == 'back':
                        cv2.circle(img, (10, 10), 10, (0, 0, 255), -1)

                    Tools.render(img[:, :, ::-1])
        except Exception as e:
            print(e)
            time.sleep(1000)
            raise e

        with open('./res/finished_files.txt', 'a+') as f:
            f.write(in_folder + '\n')

    def _store_imgs(self, buffer_, folder, start, end):
        """
            store pointer to selected images in output folder

            Args:
                buffer_: list   - buffer containing video frames
                folder:  string - path to output folder
                start:   int    - first frame of segment
                end:     int    - last frame of segment
        """
        try:
            for img_number, ii in enumerate(range(start, end + 1)):
                # make fake pointer file for speed
                # we retrieve the actual frames later while AFK
                img = np.zeros((2, 2), dtype=np.uint8)

                cv2.imwrite(
                    os.path.join(folder, str(ii).zfill(5) + '.jpg'),
                    img)
        except Exception as e:
            print(e)
            time.sleep(1000)
            raise e

    def _makedirs(self, root):
        """
            create folder structure
        """
        self._write(root)
        try:
            Tools.makedirs(root, delete=True)
            Tools.makedirs(os.path.join(root, 'left'))
        except Exception as e:
            print(e)
            time.sleep(1000)
            raise e

    def _load_videos(self, path):
        """
            read frames of left color perspective to working memory

            Args:
                path: string - path to video folder

            Returns:
                frame_buffers: list of list containing frames of videos as cv2
                               imgs
        """
        vidnames = (
            'color-recording-left.avi',)

        frame_buffers = []
        for vidname in Tools.tqdm_pbar(vidnames, "LOADING " + path):
            frame_buffers.append(self._load_video(os.path.join(path, vidname)))

        return frame_buffers

    def _load_video(self, path):
        """
            load frames of video into working memory

            Args:
                path: string - path to video file

            Returns:
                frames: list of cv2 imgs
        """

        if not os.path.isfile(path):
            path = path.replace('.avi', '-x265.mp4')

        cap = cv2.VideoCapture(path)
        frames = []
        if(not cap.isOpened()):
            print("error opening videocapture")

        ii = 0
        # Read until video is completed
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret is True:
                # store frame
                frames.append(np.copy(frame))
                ii += 1

            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()

        return frames

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
        contents = self.stdscr.instr()
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.endwin()
        print(contents)
