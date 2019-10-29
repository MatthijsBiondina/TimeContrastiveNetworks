import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from src.utils.tools import Tools


class Attention(nn.Module):
    def __init__(self, in_channels=32, beta=1., state_dict_path=None):
        """ Seq2Seq attention

        Args:
            in_channels:     int - shape of embedding space
            beta:            float - temperature of softmax function
            state_dict_path: string - path to .pth (init new random if None)
        """
        super(Attention, self).__init__()
        self.beta = beta

        # 1x1 convolutions for efficient FC layers
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels * 4, (1, 1))
        self.conv2 = nn.Conv2d(in_channels * 4, 1, (1, 1))

        if not self._load(path=state_dict_path):
            self._init_weights()

    def _load(self, path=None):
        if path is not None and os.path.isfile(path):
            try:
                if torch.cuda.is_available():
                    self.load_state_dict(torch.load(path))
                else:
                    # explicitly load to CPU
                    self.load_state_dict(torch.load(
                        path, map_location=lambda storage, loc: storage))
                Tools.log("Load Attention: Success")
                return True
            except Exception as e:
                Tools.log("Load Attention: Fail " + e)
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
            Tools.log("Init Attention: Success")
        except Exception as e:
            Tools.log("Init Attention: Fail... (abort) " + e)
            sys.exit(1)

    def forward(self, inputs):
        """ Forward pass

        Args:
            inputs: tuple() - (    (encoder_hidden_state,
                                    encoder_output_state),
                                decoder_hidden_state
                              )
        Returns:
            hidden_attention: FloatTensor - weighted mean of encoder hidden
            ouput_attention:  FloatTensor - weighted mean of encoder output
        """
        A, x = inputs
        x = x.unsqueeze(2).unsqueeze(3)
        a, o = A

        h = x.expand_as(a)
        h = torch.cat((a, h), dim=1)

        # 1x1 convolution is equivalent to fully connected layer but does not
        # require reshape of encoder outputs
        h = self.conv1(h)
        h = torch.tanh(h)

        h = self.conv2(h)
        h = F.softmax(h / self.beta, dim=3)
        h_a = torch.cat((h,) * a.shape[1], dim=1)
        h_o = torch.cat((h,) * o.shape[1], dim=1)
        a = h_a * a  # attention vector over hidden state
        o = h_o * o  # attention vector over output state

        return a.sum(dim=3).squeeze(2), o.sum(dim=3).squeeze(2)
