import random
import numpy as np
import torch
import time


class Config:
    """TOOLS"""
    TQDM = True
    PRINTLOG = False
    SAVELOG = True
    SAVERENDER = False
    SHOWRENDER = True
    SERVER = torch.cuda.is_available()  # local pc does not have cuda

    """UNIVERSAL"""
    np.random.seed(42 + int(time.time()))
    random.seed(42 + int(time.time()))
    null = torch.FloatTensor()
    N_WORKERS = 32

    """SINESET"""
    SINE_BATCH = 1
    SINE_MOMENTUM = 0.
    SINE_DIMS = 3
    SINE_EMB_DIMS = 3
    SINE_FPS = 48
    SINE_INIT_NR = 1024
    SINE_MAX_DIST = 0.1
    SINE_MAX_SPEED = 0.1
    SINE_NR_ENCODER_BLOCKS = 7
    SINE_T_MAX = 128
    SINE_T_MAX_SIM = 128

    """NAF"""
    NAF_GAMMA = 0.99
    NAF_TAU = 0.001
    NAF_BATCH_SIZE = 128
    NAF_UPDATES_PER_STEP = 5
    NAF_REPLAY_SIZE = 1000000

    """DDPG"""
    DDPG_GAMMA = 0.99
    DDPG_TAU = 0.001
    DDPG_BATCH_SIZE = 128 if torch.cuda.is_available() else 12
    DDPG_UPDATES_PER_STEP = 5
    DDPG_REPLAY_SIZE = 1000000

    """TCN"""
    TCN_IMG_SIZE = (224, 224)
    TCN_INCEPTION_FIXED_LAYERS = 50
    TCN_EMB_SIZE = 32
    TCN_BATCH = 32
    TCN_POS_RANGE = 3
    TCN_NEGATIVE_MULTIPLIER = 2
    TCN_MARGIN = 0.2
    TCN_WORKERS = 4

    """FOLDING"""
    FOLD_FPS = 12
    FOLD_MIN_LEN = 60  # threshold in seconds

    """ALPHAPOSE"""
    ALPHA_DATASET = 'coco'
    ALPHA_SP = False
    ALPHA_DETBATCH = 10
    ALPHA_FAST_INFERENCE = True
    ALPHA_POSEBATCH = 80
    ALPHA_PROFILE = False
    ALPHA_VIS_FAST = False
    ALPHA_FORMAT = 'coco'
    ALPHA_NSTACK = 4
    ALPHA_OUTPUTRESH = 80
    ALPHA_OUTPUTRESW = 64
    ALPHA_INPUTRESH = 320
    ALPHA_INPUTRESW = 256
    ALPHA_INP_DIM = 608
    ALPHA_CONFIDENCE = 0.05
    ALPHA_NUM_CLASSES = 80
    ALPHA_NMS_THESH = 0.6
    ALPHA_SAVE_IMG = False
    ALPHA_SAVE_VIDEO = False
    ALPHA_VIS = False
    ALPHA_MATCHING = False
    ALPHA_NCLASSES = 33

    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    CYAN = (255, 255, 0)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 165, 255)
    PURPLE = (255, 0, 255)

    RAD_PER_PIX = 2.74020951070339e-3

    # CAMERA AFFINE TRANSFORMATION MATRICES
    CAM_M_LEFT = np.array(
        [
            [0, 0, 0, -91],
            [0, 0, 0, 150],
            [0, 0, 0, 120],
            [0, 0, 0, 1]
        ])

    # X_l = [[283, 227, 92], [28, 182, 103], [310, 39, 127],
    #        [116, 287, 85], [5, 190, 78], [0, 0, 0]]
    # Y_l = [[0, 0, 78], [-106, -1, -6], [106, -1, -6],
    #        [-70, 0, 78], [-106, 31, 30], [-96, 144, 120]]

    # m, b = np.polynomial.polynomial.polyfit(X_l, Y_l, 1)

    XYZ_TRANSFORM = {'left_w': np.array([[0.507, -0.402, 1.476],
                                         [-0.017, -0.153, -0.878],
                                         [0.212, 0.187, -1.515]]),
                     'left_b': np.array([-96, 144, 120])}

    @staticmethod
    def crop_vals(vid_name):
        if 'color' in vid_name:
            if 'left' in vid_name:
                return {'n': 0, 's': 720, 'w': 312, 'e': 1145}
            elif 'middle' in vid_name:
                return {'n': 0, 's': 720, 'w': 290, 'e': 1036}
            elif 'right' in vid_name:
                return {'n': 0, 's': 720, 'w': 165, 'e': 963}
        elif 'depth' in vid_name:
            if 'left' in vid_name:
                return {'n': 14, 's': 406, 'w': 88, 'e': 512}
            elif 'middle' in vid_name:
                return {'n': 18, 's': 395, 'w': 60, 'e': 452}
            elif 'right' in vid_name:
                return {'n': 0, 's': 406, 'w': 0, 'e': 424}
        print('Error cropping', vid_name)
