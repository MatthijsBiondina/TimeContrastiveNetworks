import cv2
import numpy as np
import random
from random import uniform
import torchvision

from src.utils.tools import Tools
from src.utils.config import Config


class Transformer:
    @staticmethod
    def transform(img,
                  augment=False,
                  BGR=True,
                  d_hue=None,
                  d_sat=None,
                  d_val=None,
                  d_con=None):
        """
            transform image

            @PARAMS:
            img := opencv bgr img - image to transform
            augment := bool - augment img if True, o/w only inception prep
            d_hue   := [-1,1] - change to hue;        None -> random
            d_sat   := [-1,1] - change to saturation; None -> random
            d_val   := [-1,1] - change to value;      None -> random
            d_con   := [-1,1] - change to contrast;   None -> random
        """
        if not BGR:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if augment:
            # select random factors if unspecified
            d_hue = d_hue if d_hue is not None else uniform(-1, 1)
            d_sat = d_sat if d_sat is not None else uniform(-1, 1)
            d_val = d_val if d_val is not None else uniform(-1, 1)
            d_con = d_con if d_con is not None else uniform(-1, 1)

            # convert to hsv
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img = np.array(img, dtype=int)

            # change hsv
            img[:, :, 0] = img[:, :, 0] + int(d_hue * 32)  # hue
            img[:, :, 1] = img[:, :, 1] + int(d_sat * 32)  # saturation
            img[:, :, 2] = img[:, :, 2] + int(d_val * 32)  # value
            img = np.clip(img, 0, 255).astype(np.uint8)

            # change contrast
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(img)
            clahe = cv2.createCLAHE(
                clipLimit=1 + 0.9 * d_con, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            img = cv2.merge((cl, a, b))
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img[:, :, :3]

        if augment:
            max_side_len = min(img.shape[0], img.shape[1])
            side_len = random.randint(Config.TCN_IMG_SIZE[0], max_side_len)

            x_l = random.randint(0, img.shape[1] - side_len)
            y_l = random.randint(0, img.shape[0] - side_len)
            img = img[y_l:y_l + side_len, x_l:x_l + side_len, :]

        img = cv2.resize(img, Config.TCN_IMG_SIZE).astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # img -= np.mean(np.mean(img, axis=1, keepdims=True),
        #                axis=0, keepdims=True)

        # img /= np.std(np.std(img, axis=1, keepdims=True),
        #               axis=0, keepdims=True)

        # Tools.pyout(img)
        # Tools.exit()

        # # img -= np.mean(img, 2).expand_as(img)

        # Tools.render(Transformer.untransform_np(img), waitkey=1000)

        # img = img.transpose(2, 0, 1)
        # img = img.astype(np.float32)
        # img = (img / 255 - 0.5) * 2.0

        return img

    @staticmethod
    def untransform_np(img):
        img += np.min(img)
        img /= np.max(img)
        img = img * 255
        img = img.astype(np.uint8)
        img = img.transpose(1, 2, 0)
        return img

    @staticmethod
    def untransform(tensor):
        img = (tensor + 1.0) / 2 * 255
        img = img.transpose(1, 2, 0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
