import os

from src.utils.tools import Tools


class GatherTrials:
    """
        original raw footage is in unstructured folder tree. Find all valid trials and store in text file for later use
    """
    def __init__(self, root):
        self.root = root

    def walk_dirs(self, ou_path):
        """
            find trials and store

            Args:
                ou_path: string - path to output file
        """
        recording_paths = []

        for f0 in Tools.tqdm_pbar(Tools.list_dirs(self.root), "GATHERING"):
            fname = Tools.fname(f0)
            if fname == '2017-11-09-to-encode':
                for f1 in Tools.list_dirs(f0):
                    for f2 in Tools.list_dirs(f1):
                        if self._valid_folder(f2):
                            if self._length(f2, encoded=False) >= 60:
                                recording_paths.append(f2)
            elif fname == '2017-11-06-still some parts to compress':
                for f1 in Tools.list_dirs(f0):
                    if self._valid_folder(f1):
                        if self._length(f1, encoded=False) >= 60:
                            recording_paths.append(f1)
            elif fname == 'toencode':
                for f1 in Tools.list_dirs(f0):
                    fname1 = Tools.fname(f1)
                    if "copy" not in fname1 and not fname1 == "encoded":
                        for f2 in Tools.list_dirs(f1):
                            if self._valid_folder(f2):
                                if self._length(f2, encoded=False) >= 60:
                                    recording_paths.append(f2)
            else:
                for f1 in Tools.list_dirs(f0):
                    if self._valid_folder(f1):
                        if self._length(f1) >= 60:
                            recording_paths.append(f1)

        with open(ou_path, 'w+') as f:
            f.write('\n'.join(recording_paths))

    def _valid_folder(self, path):
        """
            check whether a folder contains all video files to qualify as a
            valid video folder

            Args:
                path: string - path to directory

            Returns:
                True if folder contains all three color and all three depth
                        perspectives
                False o/w
        """
        if not os.path.isdir(path):
            print("not a dir " + path)
            return False
        if not (os.path.isfile(
                os.path.join(path, 'color-recording-left.avi')) or
                os.path.isfile(
                os.path.join(path, 'color-recording-left-x265.mp4'))):
            return False
        if not (os.path.isfile(
            os.path.join(path, 'color-recording-middle.avi')) or
                os.path.isfile(
                    os.path.join(path, 'color-recording-middle-x265.mp4'))):
            return False
        if not (os.path.isfile(
            os.path.join(path, 'color-recording-right.avi')) or
                os.path.isfile(
                    os.path.join(path, 'color-recording-right-x265.mp4'))):
            return False
        if not (os.path.isfile(
            os.path.join(path, 'depth-recording-left.avi')) or
                os.path.isfile(
                    os.path.join(path, 'depth-recording-left-x265.mp4'))):
            return False
        if not (os.path.isfile(
            os.path.join(path, 'depth-recording-middle.avi')) or
                os.path.isfile(
                    os.path.join(path, 'depth-recording-middle-x265.mp4'))):
            return False
        if not (os.path.isfile(
            os.path.join(path, 'depth-recording-right.avi')) or
                os.path.isfile(
                    os.path.join(path, 'depth-recording-right-x265.mp4'))):
            return False

        return True

    def _length(self, path, encoded=True):
        """
            get lenght of trial in folder. Recordings of less than 1 minute
            are ignored.

            Args:
                path: string - path to trial root folder
                encoded: bool - whether video is encoded (do manually, o/w use
                                metadata)
            Returns:
                float: lenght of video in seconds
        """
        if encoded:
            if os.path.isfile(os.path.join(
                    path, 'depth-recording-left.avi')):
                vidpath = os.path.join(path, 'depth-recording-left.avi')
            else:
                vidpath = os.path.join(path, 'depth-recording-left-x265.mp4')
            return Tools.video_length(vidpath)
        else:
            return int(os.path.getsize(
                os.path.join(path, 'depth-recording-left.avi')) / 1e6)
