import cv2
import numpy as np
import os
import shutil
import sys
import time
import traceback

from src.utils.config import Config


class Tools:
    """
        Commonly used methods and variations on standard python methods
    """

    @staticmethod
    def fname(path):
        """
            get name of directory of file without path towards it

            Args:
                path: string - path to file or directory

            Returns:
                string - name of folder of file
        """
        if True or os.path.exists(path):
            return str(path.split('/')[-1])
        else:
            raise FileNotFoundError

    @staticmethod
    def list_dirs(folder):
        """
            list directories in directory

            Args:
                folder: string - path to directory

            Returns:
                list - paths to directories in input directory
        """
        return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                       if os.path.isdir(os.path.join(folder, f))])

    @staticmethod
    def list_files(folder, end=None, substr=None):
        """
            list files in a directory

            Args:
                folder: string - path to directory
                end:    string - only return files that end with this string
                substr: string - only return files with substr in filename

            Returns:
                list of filepaths
        """
        return sorted(
            [os.path.join(folder, f) for f in os.listdir(folder) if
             os.path.isfile(os.path.join(folder, f)) and not
             os.path.isdir(os.path.join(folder, f)) and
             (end is None or f.endswith(end)) and
             (substr is None or substr in f)])

    @staticmethod
    def makedirs(folder, delete=False):
        """
            Create new directory.

            Args:
                folder: str     - path to new folder
                delete: boolean - if true and folder already exists, delete
                                  contents

        """
        try:
            os.makedirs(folder)
        except FileExistsError:
            if delete:
                shutil.rmtree(folder)
                os.makedirs(folder)

    @staticmethod
    def pyout(*args, force=True, love_u=Config.TQDM, end='\n'):
        """
            print using tqdm.write

            Args:
                *args: *string - to print
                force: bool - ignore config settings
                end: string - append to end of line
        """
        if Config.PRINTLOG or force:
            print(datetime.datetime.now().strftime(
                "%d-%m-%Y %H:%M"), *args, end=end)
        elif love_u:
            tqdm.write(datetime.datetime.now().strftime(
                "%d-%m-%Y %H:%M"), end=' ')
            for arg in list(map(str, args)):
                for ii, string in enumerate(arg.split('\n')):
                    if ii == len(arg.split('\n')) - 1:
                        tqdm.write(string, end=' ')
                    else:
                        tqdm.write(string)
            tqdm.write('', end=end)

    @staticmethod
    def render(img, name='DEBUG', waitkey=50, ascii=False, curse=None):
        """
            render image to screen

            Args:
                img: cv2 - image
                name: string - window screen
                waitkey: int - ms to sleep after render
                ascii: bool - display as ascii art instead of cv2 render
                curse: curse - display ascii art with curse
        """
        if ascii or curse:
            chars = np.asarray(list(' .,:;irsXA253hMHGS#9B&@'))
            img = cv2.resize(img, (178, 55))
            while len(img.shape) > 2:
                img = np.mean(img, axis=2)
            img = (img - np.min(img)) / \
                max(1., np.max(img) - np.min(img)) * (len(chars) - 1)
            img = np.clip(img, 0, len(chars) - 1)
            # print('\x1bc')
            #
            rows = ("".join(r) for r in chars[img.astype(int)])

            if ascii:
                print("\n".join(rows))
                print('\033[58A', end='')
            else:
                for row_ii, row in enumerate(rows):
                    curse.addstr(row_ii + 1, 1, row)
            time.sleep(waitkey / 1000)
        else:
            exit = False
            try:
                cv2.imshow(name, img)
                if cv2.waitKey(waitkey) & 0xFF == ord('q'):
                    exit = True
            except Exception:
                Tools.pyout('dropped frame')
                Tools.debug(traceback.print_exc())
            if exit:
                sys.exit(0)

    @staticmethod
    def tqdm_pbar(iterable, description="DEBUG", total=None):
        """
            named progress bar

            Args:
                iterable:    iterable - list to loop over
                description: string   - displayed at start of line before bar
                total:       int      - number of items in iterable if this
                                        cannot be determined automatically

            Returns:
                tqdm.pbar
        """
        pbar = tqdm(iterable, total=total)
        if len(description) <= 23:
            pbar.set_description(description + ' ' * (23 - len(description)))
        else:
            pbar.set_description(description[:20] + '...')
        return pbar

    @staticmethod
    def video_length(fileloc):
        """
            get length of video from emtadata

            Args:
                fileloc: string - path to video file

            Returns:
                float - length in seconds
        """
        command = ['ffprobe',
                   '-v', 'fatal',
                   '-show_entries',
                   'stream=width,height,r_frame_rate,duration',
                   '-of', 'default=noprint_wrappers=1:nokey=1',
                   fileloc, '-sexagesimal']
        ffmpeg = subprocess.Popen(
            command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = ffmpeg.communicate()
        if(err):
            print(err)
        out = out.decode('utf-8').split('\n')
        try:
            return int(out[3].split(':')[1]) * 60 + \
                int(float(out[3].split(':')[2]))
        except IndexError:
            return -float("inf")
