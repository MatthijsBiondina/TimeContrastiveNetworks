import colorsys
import cv2
import datetime
import json
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
from PIL import ImageEnhance
import random
from random import shuffle
from random import randint
import shutil
from subprocess import DEVNULL, STDOUT, check_call
import sys
import time
import torch
from tqdm import tqdm
import traceback

from src.utils.config import Config


class Tools:

    @staticmethod
    def log(message):
        """log time and message

        Args:
          message: string - to print/write to file
        Returns:
          n/a
        """
        if Config.PRINTLOG:
            Tools.pyout(message)
        if Config.SAVELOG:
            with open('log.txt', 'a+') as f:
                f.write('\n' +
                        datetime.datetime.now().strftime("%d-%m-%Y %H:%M") +
                        ' - ' +
                        str(message) +
                        '\n')

    @staticmethod
    def pyout(*args, force=False, love_u=Config.TQDM, end='\n'):
        """print if printing is ON

        Args:
            *args: strings - to print
            force: bool - ignore config setting
            end:    string   - append to end of line
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
    def debug(*args, force=False, love_u=Config.TQDM, end='\n', ex=None):
        Tools.pyout(
            "DEBUG ->", traceback.format_stack()[-2].split('\n')[0])
        Tools.pyout("------->  ", *args, force=force, love_u=love_u, end=end)
        if ex is not None:
            sys.exit(ex)

    @staticmethod
    def tqdm_pbar(iterable, description="DEBUG", total=None):
        pbar = tqdm(iterable, total=total)
        if len(description) <= 23:
            pbar.set_description(description + ' ' * (23 - len(description)))
        else:
            pbar.set_description(description[:20] + '...')
        return pbar

    @staticmethod
    def soft_update(target, source, tau):
        """update parameters of target network with incremental step towards
           source network

        Args:
          target: torch.nn.Module - target network
          source: torch.nn.Module - source network
          tau:    float - step size
        Returns:
          n/a
        """
        for target_param, param in zip(target.parameters(),
                                       source.parameters()):
            target_param.data.copy_(
                target_param.data * (1. - tau) + param.data * tau)

    @staticmethod
    def hard_update(target, source):
        """update parameters of target network to source network

        Args:
          target: torch.nn.Module - target network
          source: torch.nn.Module - source network
        Returns:
          n/a
        """
        for target_param, param in zip(target.parameters(),
                                       source.parameters()):
            target_param.data.copy_(param.data)

    @staticmethod
    def contains_nan(tensor):
        return torch.sum(tensor != tensor) > 0

    @staticmethod
    def nearest_neighbor(target, source):
        """get element of target that is nearest to source

        Args:
            target: trajectory
            source: point
        """
        distances = target - source.unsqueeze(3).expand_as(target)
        distances = torch.sum(distances.abs_(), dim=1)
        _, argmin = distances.min(-1)
        argmin = argmin.squeeze().cpu().numpy()
        return target[
            np.arange(target.shape[0]), :, :, argmin.squeeze()].squeeze(2)

    @staticmethod
    def pdist(x):
        n_obs, dim = x.size()
        xi = x.unsqueeze(0).expand(n_obs, n_obs, dim)
        xj = x.unsqueeze(1).expand(n_obs, n_obs, dim)
        dij = ((xi - xj)**2.).sum(2).squeeze()
        return dij

    @staticmethod
    def pequal(x):
        n_obs, = x.size()
        xi = x.unsqueeze(0).expand(n_obs, n_obs)
        xj = x.unsqueeze(1).expand(n_obs, n_obs)
        eij = (xi == xj)
        return eij

    @staticmethod
    def transform(img, augment=False):
        if augment:
            ops = [
                'ImageEnhance.Brightness(img).enhance(np.random.uniform('
                'low=1 - 32 / 255, high=1 + 32 / 255))',  # brightness
                'ImageEnhance.Color(img).enhance('
                'np.random.uniform(low=0.5, high=1.5))',  # saturation
                'ImageEnhance.Contrast(img).enhance('
                'np.random.uniform(low=0.5,high=1.5))',  # contrast
                'Image.fromarray(np.uint8(np.array(img.convert("HSV")) +'
                '(np.random.uniform(-50,50),0,0)), mode="HSV").convert("RGB")'
            ]  # hue
            shuffle(ops)
            for op in ops:
                img = eval(op)

        width, height = img.size
        left = randint(0, width - 299)
        upper = randint(0, height - 299)
        right = left + 299
        lower = upper + 299
        img = img.crop((left, upper, right, lower))

        return (np.array(
            img, dtype=np.float32).transpose(2, 0, 1) / 255. - 0.5) * 2.

    @staticmethod
    def list_dirs(folder):
        return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                       if os.path.isdir(os.path.join(folder, f))])

    @staticmethod
    def list_files(folder, end=None, substr=None):
        return sorted(
            [os.path.join(folder, f) for f in os.listdir(folder) if
             os.path.isfile(os.path.join(folder, f)) and not
             os.path.isdir(os.path.join(folder, f)) and
             (end is None or f.endswith(end)) and
             (substr is None or substr in f)])

    @staticmethod
    def makedirs(folder, delete=False):
        try:
            os.makedirs(folder)
        except FileExistsError:
            if delete:
                shutil.rmtree(folder)
                os.makedirs(folder)

    @staticmethod
    def search(folder, filenames, substr=False):
        retlst = []
        for root, dirs, files in os.walk(folder, topdown=True):
            for f_name in sorted(files):
                if substr:
                    if any([filename in f_name for filename in filenames]):
                        retlst.append(os.path.join(root, f_name))
                else:
                    if f_name in filenames:
                        retlst.append(os.path.join(root, f_name))
        return retlst

    @staticmethod
    def search_dir(search_folder, folder_name):
        retlst = []
        for root, dirs, _ in os.walk(search_folder, topdown=True):
            for dir_ in dirs:
                if dir_ == folder_name:
                    retlst.append(os.path.join(root, dir_))
        return retlst

    @staticmethod
    def fname(path):
        if True or os.path.exists(path):
            return str(path.split('/')[-1])
        else:
            raise FileNotFoundError

    @staticmethod
    def pathstr(path):
        return path.replace('(', '\\(').replace(')', '\\)').replace(' ', '\\ ')

    @staticmethod
    def shuffled(lst):
        return sorted(lst, key=lambda x: random.random())

    @staticmethod
    def time(time1=0):
        if not time1:
            return time.time()
        else:
            interval = time.time() - time1
            return time.time(), interval

    @staticmethod
    def exit(code=0):
        print("\n\n\n\nIt's...\n")
        sys.exit(code)

    @staticmethod
    def readable_json():
        for path, subdirs, files in os.walk('.'):
            for f in files:
                if f.endswith('.json'):
                    try:
                        with open(os.path.join(path, f), 'r') as j:
                            d = json.load(j)

                        with open(os.path.join(path, f), 'w') as j:
                            json.dump(d, j, indent=2)
                        print(os.path.join(path, f))
                    except json.decoder.JSONDecodeError:
                        pass

    @staticmethod
    def line(P1, P2):
        P1, P2 = np.array(P1).astype(float), np.array(P2).astype(float)
        a = (P2[1] - P1[1]) / (P2[0] - P1[0])
        b = P1[1] - a * P1[0]
        return (a, b)

    def line_x(L, x):
        return L[0] * x + L[1]

    @staticmethod
    def num2hslstr(t):
        cmap = plt.cm.get_cmap('gist_rainbow')
        rgb = cmap(1 - t)[:3]
        hls = colorsys.rgb_to_hls(*rgb)
        return 'hsl({},{}%,{}%)'.format(
            str(int(hls[0] * 360)),
            str(int(hls[2] * 100)),
            str(int(hls[1] * 100)))

    @staticmethod
    def handbrake(in_file):
        assert '.mp4' in in_file
        cp_file = in_file.replace('.mp4', '_copy.mp4')
        os.rename(in_file, cp_file)

        check_call([
            'HandBrakeCLI', '-Z', 'Universal', '-i', cp_file, '-o', in_file],
            stdout=DEVNULL, stderr=STDOUT)
        os.remove(cp_file)

    @staticmethod
    def ffmpeg(in_folder):
        for vid_file in Tools.tqdm_pbar(
                Tools.list_files(in_folder, end='.mp4'), 'CONVERTING VIDEOS'):
            os.rename(vid_file, os.path.join(in_folder, 'foobar.mp4'))

            check_call([
                'HandBrakeCLI',
                '-Z',
                'Universal',
                '-i',
                os.path.join(in_folder, 'foobar.mp4'),
                '-o',
                vid_file], stdout=DEVNULL, stderr=STDOUT)

            # check_call([
            #     'ffmpeg', '-i',
            #     os.path.join(in_folder, 'foobar.mp4'),
            #     vid_file], stdout=DEVNULL,
            #     stderr=STDOUT)
            os.remove(os.path.join(in_folder, 'foobar.mp4'))

    @staticmethod
    def line_fn(P1, P2):
        a, b = Tools.line(P1, P2)
        return lambda x: a * x + b

    @staticmethod
    def line_intersection(L1, L2):
        try:
            a1, b1 = L1
            a2, b2 = L2
            x = (b2 - b1) / (a1 - a2)
            y = a1 * x + b1
            assert(x == x and y == y)  # isnan
            return (x, y)
        except Exception as a:
            print(L1, L2)
            raise a

    @staticmethod
    def gray2cm(x):
        return 0.61 * x + 160

    @staticmethod
    def clip(val, minval=-float('inf'), maxval=float('inf')):
        return max(minval, min(maxval, val))

    @staticmethod
    def render(img, name='DEBUG', ascii=False, curse=None, waitkey=50):
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
        elif True:  # not Config.SERVER:
            exit = False
            try:
                cv2.imshow(name, img)
                if cv2.waitKey(waitkey) & 0xFF == ord('q'):
                    exit = True
            except Exception as e:
                Tools.pyout('dropped frame')
                Tools.debug(traceback.print_exc())
            if exit:
                sys.exit(0)

    @staticmethod
    def fig2img(fig):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA
        channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode.
        # Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        return buf
