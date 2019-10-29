import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

from src.utils.tools import Tools


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, dilation=None):
        """ One residual block from the Wavenet architecture.
            No skip-connections are used because we use attention.

        Args:
            in_channels: int - must be divisible by 2
            dilation:    int - should increase exponentially to widen fov
        """
        super(ResidualBlock, self).__init__()
        assert(in_channels % 2 == 0)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # dilated convolution to the left (past)
        self.padd1_l = nn.ZeroPad2d((dilation, 0, 0, 0))
        self.dcnv1_l = nn.Conv2d(
            in_channels // 2, in_channels, (1, 2), dilation=dilation)

        # dilated convolution to the right (future)
        self.padd1_r = nn.ZeroPad2d((0, dilation, 0, 0))
        self.dcnv1_r = nn.Conv2d(in_channels // 2,
                                 in_channels,
                                 (1, 2),
                                 dilation=dilation)

        self.conv2_skp = nn.Conv2d(in_channels, out_channels, (1, 1))
        self.conv2_hid = nn.Conv2d(in_channels, in_channels, (1, 1))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights to very small values so that initially, ResidualBlocks
           output the identity function.
        """
        # dcnv1_l
        nn.init.xavier_normal_(self.dcnv1_l.weight.data, gain=1e-8)
        nn.init.normal_(self.dcnv1_l.bias.data, std=1e-8)

        # dcnv1_r
        nn.init.xavier_normal_(self.dcnv1_r.weight.data, gain=1e-8)
        nn.init.normal_(self.dcnv1_r.bias.data, std=1e-8)

        # conv2_hid
        nn.init.xavier_normal_(self.conv2_hid.weight.data, gain=1e-8)
        nn.init.normal_(self.conv2_hid.bias.data, std=1e-8)

        # conv2_skp
        nn.init.xavier_normal_(self.conv2_skp.weight.data)
        nn.init.normal_(self.conv2_skp.bias.data)

    def forward(self, x):
        h_l, h_r = torch.split(x, x.shape[1] // 2, dim=1)

        h_l = self.padd1_l(h_l)
        h_l = self.dcnv1_l(h_l)

        h_r = self.padd1_r(h_r)
        h_r = self.dcnv1_r(h_r)

        # split for gated activation
        h_lf, h_lg = torch.split(h_l, h_l.shape[1] // 2, dim=1)
        h_rf, h_rg = torch.split(h_r, h_r.shape[1] // 2, dim=1)

        h_f = torch.cat((h_lf, h_rf), dim=1)
        h_g = torch.cat((h_lg, h_rg), dim=1)

        h = torch.tanh(h_f) * torch.sigmoid(h_g)

        # 1x1 convolution as an efficient FC operation
        o = self.conv2_skp(h)
        h = self.conv2_hid(h)

        return x + h, o  # residual connection


class Encoder(nn.Module):
    def __init__(self,
                 in_channels=32,
                 out_channels=32,
                 nr_blocks=7,
                 device=None,
                 state_dict_path=None):
        """ Seq2Seq encoder using Wavenet architecture

        Args:
            in_channels:     int - shape of embedding space
            nr_blocks:       int - nr of residual blocks
                                   (dilation = 2**(nr_blocks+1))
            device:          torch.device
            state_dict_path: string - path to .pth
        """
        super(Encoder, self).__init__()
        self.out_channels = out_channels
        self.device = device

        self.residual_blocks = []
        for i in range(nr_blocks):
            self.residual_blocks.append(
                ResidualBlock(in_channels=in_channels * 2,
                              out_channels=out_channels * 2,
                              dilation=2**i).to(device))

        self.residual_blocks = nn.Sequential(*self.residual_blocks)

        self.conv1 = nn.Conv2d(out_channels * 2, out_channels * 2, (1, 1))
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, (1, 1))

        if not self._load(path=state_dict_path):
            self._init_weights()

    def _load(self, path):
        if path is not None and os.path.isfile(path):
            try:
                if torch.cuda.is_available():
                    self.load_state_dict(torch.load(path))
                else:
                    self.load_state_dict(torch.load(
                        path, map_location=lambda storage, loc: storage))
                Tools.log("Load Encoder: Success")
                return True
            except Exception as e:
                Tools.log("Load Encoder: Fail " + e)
                return False
        return False

    def _init_weights(self):
        try:
            # conv1
            nn.init.xavier_normal_(self.conv1.weight.data)
            nn.init.normal_(self.conv1.bias.data)

            # conv2
            nn.init.xavier_normal_(self.conv2.weight.data)
            nn.init.normal_(self.conv2.bias.data)
            Tools.log("Init Encoder: Success")
        except Exception as e:
            Tools.log("Init Encoder: Fail... (abort) " + e)
            sys.exit(1)

    def forward(self, x):
        h = torch.cat((x, x), dim=1)

        o = torch.zeros((x.shape[0],
                         self.out_channels * 2,
                         x.shape[2], x.shape[3])).to(self.device)

        for block in self.residual_blocks:
            h, o_ = block(h)
            o += o_

        o = F.relu(o)
        o = self.conv1(o)
        o = F.relu(o)
        o = self.conv2(o)

        return h, o
