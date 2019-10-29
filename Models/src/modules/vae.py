import os
import sys
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from src.utils.tools import Tools


class VAE(nn.Module):
    def __init__(self,
                 in_channels=3,
                 embedding_dim=512,
                 state_dict_path=None):
        super(VAE, self).__init__()

        # ENCODER
        # 224 x 224 pixels
        self.conv1 = nn.Conv2d(in_channels, 32, (9, 9), stride=3, padding=0)
        self.conv2 = nn.Conv2d(32, 64, (7, 7), stride=3, padding=0)
        self.conv3 = nn.Conv2d(64, 512, (22, 22))

        # BOTTLENECK
        self.line_mu = nn.Linear(512, embedding_dim)
        self.line_logvar = nn.Linear(512, embedding_dim)

        # DECODER
        self.line4 = nn.Linear(embedding_dim, 512)

        self.trans5 = nn.ConvTranspose2d(512, 128, (28, 28))
        self.conv5 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.trans6 = nn.ConvTranspose2d(128, 64, (8, 8), stride=4, padding=2)
        self.conv6 = nn.Conv2d(64, 64, (3, 3), padding=1)

        self.trans7 = nn.ConvTranspose2d(
            64, in_channels, (4, 4), stride=2, padding=2)
        self.conv7 = nn.Conv2d(in_channels, in_channels, (9, 9), padding=5)

        if not self._load(path=state_dict_path):
            # self._init_weights()
            pass

    def _load(self, path=None):
        if path is not None and os.path.isfile(path):
            try:
                if torch.cuda.is_available():
                    self.load_state_dict(torch.load(path))
                else:
                    self.load_state_dict(torch.load(
                        path, map_location=lambda storage, loc: storage))
                Tools.log("Load VAE: Success")
                return True
            except Exception as e:
                Tools.pyout("Load VAE: Fail " + e)
                Tools.log("Load VAE: Fail " + e)
                return False
        return False

    def _init_weights(self):
        try:
            nn.init.xavier_normal_(self.conv1.weight)
            nn.init.normal_(self.conv1.bias)

            nn.init.xavier_normal_(self.conv2.weight)
            nn.init.normal_(self.conv2.bias)

            nn.init.xavier_normal_(self.conv3.weight)
            nn.init.normal_(self.conv3.bias)

            nn.init.xavier_normal_(self.line_mu.weight)
            nn.init.normal_(self.line_mu.bias)

            nn.init.xavier_normal_(self.line_logvar.weight)
            nn.init.normal_(self.line_logvar.bias)

            nn.init.xavier_normal_(self.line4.weight)
            nn.init.normal_(self.line4.bias)

            nn.init.xavier_normal_(self.trans5.weight)
            nn.init.normal_(self.trans5.bias)

            nn.init.xavier_normal_(self.conv5.weight)
            nn.init.normal_(self.conv5.bias)

            nn.init.xavier_normal_(self.trans6.weight)
            nn.init.normal_(self.trans6.bias)

            nn.init.xavier_normal_(self.conv6.weight)
            nn.init.normal_(self.conv6.bias)

            nn.init.xavier_normal_(self.trans7.weight)
            nn.init.normal_(self.trans7.bias)

            nn.init.xavier_normal_(self.conv7.weight)
            nn.init.normal_(self.conv7.bias)
        except Exception as e:
            Tools.log('Init VAE: Fail ' + e)
            sys.exit(1)

    def encode(self, x):
        h = torch.tanh(self.conv1(x))
        h = torch.tanh(self.conv2(h))
        h = torch.relu(self.conv3(h))
        h = h.squeeze(3).squeeze(2)

        return self.line_mu(h), self.line_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            H = Normal(mu, logvar.mul(0.5).exp_())
            return H.rsample()
        else:
            return mu

    def decode(self, x):
        h = torch.relu(self.line4(x))
        h = h.unsqueeze(2).unsqueeze(3)
        h = torch.tanh(self.conv5(self.trans5(h)))
        h = torch.tanh(self.conv6(self.trans6(h)))
        h = torch.tanh(self.conv7(self.trans7(h)))

        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def save_state_dict(self, name):
        path = os.path.join('./res/models', name)
        try:
            os.makedirs(path)
        except OSError:
            pass

        torch.save(self.state_dict(), os.path.join(path, 'vae_mdl.pth'))
