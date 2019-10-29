import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.models as models
import numpy as np
import os
import sys

from src.utils.tools import Tools
from src.utils.config import Config


class SpatialSoftmax(nn.Module):
    """Chelsea Finn
    Returns softargmax over widht*height dimensions of tensor
    """

    def __init__(self,
                 height,
                 width,
                 channel,
                 temperature=None,
                 data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.channel = channel
        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width))
        pos_x = torch.from_numpy(pos_x.reshape(
            self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(
            self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        feature = feature.view(-1, self.height * self.width)
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(
            self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(
            self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints


class GPU0(nn.Module):
    def __init__(self, state_dict_path=None):
        super(GPU0, self).__init__()
        self.inception = nn.Sequential(
            *list(models.inception_v3(pretrained=True).children())[:8])
        self.inception.eval()
        for ii, param in enumerate(self.inception.parameters()):
            # if ii < Config.TCN_INCEPTION_FIXED_LAYERS:
            param.requires_grad = False

        self.drop0 = nn.Dropout2d(p=0.2)

        self.conv1 = nn.Conv2d(288, 512, (3, 3), stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.bnrm1 = nn.BatchNorm2d(512)

        if not self._load(path=state_dict_path):
            self._init_weights()

    def _load(self, path=None):
        if path is not None and os.path.isfile(path):
            try:
                if torch.cuda.is_available():
                    self.load_state_dict(torch.load(path))
                else:
                    self.load_state_dict(torch.load(
                        path, map_location=lambda storage, loc: storage))
                Tools.log("Load TCN on device 0: Success")
                return True
            except Exception as e:
                Tools.log("Load TCN on device 0: Fail " + e)
                return False
        return False

    def _init_weights(self):
        try:
            nn.init.xavier_normal_(self.conv1.weight)
            nn.init.normal_(self.conv1.bias)
            Tools.log("Init TCN on device 0: Success")
        except Exception as e:
            Tools.log("Init TCN on device 0: Fail... (abort) " + e)
            sys.exit(1)

    def forward(self, x):
        h = self.inception(x)
        h = self.drop0(h)
        h = self.conv1(h)
        h = self.relu1(h)
        h = self.bnrm1(h)
        return h


class GPU1(nn.Module):
    def __init__(self, state_dict_path=None):
        super(GPU1, self).__init__()
        self.conv2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.bnrm2 = nn.BatchNorm2d(512)
        self.spat2 = SpatialSoftmax(141, 141, 512, temperature=1.)

        self.full3 = nn.Linear(1024, 2048)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.2)

        self.full4 = nn.Linear(2048, 32)

        if not self._load(path=state_dict_path):
            self._init_weights()

    def _load(self, path=None):
        if path is not None and os.path.isfile(path):
            try:
                if torch.cuda.is_available():
                    self.load_state_dict(torch.load(path))
                else:
                    self.load_state_dict(torch.load(
                        path, map_location=lambda storage, loc: storage))
                Tools.log("Load TCN on device 1: Success")
                return True
            except Exception as e:
                Tools.log("Load TCN on device 1: Fail " + e)
                return False
        return False

    def _init_weights(self):
        try:
            nn.init.xavier_normal_(self.conv2.weight)
            nn.init.normal_(self.conv2.bias)

            nn.init.xavier_normal_(self.full3.weight)
            nn.init.normal_(self.full3.bias)

            nn.init.xavier_normal_(self.full4.weight)
            nn.init.normal_(self.full4.bias)
            Tools.log("Init TCN on device 1: Success")
        except Exception as e:
            Tools.log("Init TCN on device 1: Fail... (abort) " + e)
            sys.exit(1)

    def forward(self, x):
        h = self.conv2(x)
        h = self.relu2(h)
        h = self.bnrm2(h)
        h = self.spat2(h)

        h = self.full3(h)
        h = self.relu3(h)
        h = self.drop3(h)

        h = self.full4(h)
        h = F.normalize(h)
        return h


class TCN(nn.Module):
    def __init__(self, devices, state_dict_paths=(None, None)):
        super(TCN, self).__init__()
        self.gpu0 = GPU0(state_dict_path=state_dict_paths[0])
        self.gpu1 = GPU1(state_dict_path=state_dict_paths[1])

        self.devices = devices
        self.gpu0.to(self.devices[0])
        self.gpu1.to(self.devices[1])

    def forward(self, x):
        h = self.gpu0(x)
        h = h.to(self.devices[1])
        h = self.gpu1(h)
        return h

    def save_state_dicts(self, name):
        path = os.path.join('./res/models', name)
        try:
            os.makedirs(path)
        except OSError:
            pass
        torch.save(self.gpu0.state_dict(),
                   os.path.join(path, 'gpu0.pth'))
        torch.save(self.gpu1.state_dict(),
                   os.path.join(path, 'gpu1.pth'))

    def switch_mode(self, mode='train'):
        exec('self.gpu0.drop0.' + mode + '()')
        exec('self.gpu0.conv1.' + mode + '()')
        exec('self.gpu0.bnrm1.' + mode + '()')
        exec('self.gpu1.' + mode + '()')
