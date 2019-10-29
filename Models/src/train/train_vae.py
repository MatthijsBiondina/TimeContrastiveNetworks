import sys
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.datasets.vae_set import VAESet
from src.modules.vae import VAE
from src.utils.tools import Tools
poem = Tools.tqdm_pbar


class VAETrainer:
    def __init__(self,
                 device=None,
                 state_dict_path=None,
                 train_root=None,
                 val_root=None,
                 pos=None,
                 save_loc=None):
        self.device = device
        self.state_dict_path = state_dict_path
        assert pos is not None
        self.pos = pos
        self.save_loc = save_loc

        self.train_set = VAESet(
            root_dir=train_root,
            pos=pos,
            augment=True)
        self.val_set = VAESet(
            root_dir=val_root,
            pos=pos,
            augment=False)

        self.vae = VAE().to(device)
        self.optimizer = optim.Adam(list(filter(lambda p: p.requires_grad,
                                                self.vae.parameters())),
                                    lr=1e-4)

        Tools.pyout('VAETrainer loaded')

    def loop(self, epochs):
        self.best_loss = float('inf')
        for epoch in poem(range(epochs), self.pos):
            self.train(epoch)
            self.eval(epoch)

    def train(self, epoch):
        self.vae.train()
        epoch_loss = 0.
        for ii in poem(
                range(min(500, len(self.train_set))),
                "TRAIN EPOCH " + str(epoch)):
            X, t, _ = self.train_set[ii]

            X, t = X.to(self.device), t.to(self.device)
            y, mu, logvar = self.vae(X)

            self.optimizer.zero_grad()

            BCE = F.binary_cross_entropy(
                (1.0 + y.reshape(X.shape[0], -1)) / 2,
                (1.0 + t.reshape(X.shape[0], -1)) / 2,
                reduction='sum')

            logvar = torch.clamp(logvar, -10., 10.)
            mu = torch.clamp(mu, -10., 10.)

            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            BCE /= X.shape[0] * X.shape[1] * X.shape[2] * X.shape[3]
            KLD /= X.shape[0] * X.shape[1] * X.shape[2] * X.shape[3]

            loss = BCE + KLD
            if loss < 12.0:
                loss.backward()
                self.optimizer.step()
            # else:
            #     Tools.pyout("Skip: ")

            kld = KLD.item()
            # Tools.pyout(BCE.item(), KLD.item())
            if kld != kld:
                sys.exit(0)

            epoch_loss += loss.item()

        Tools.pyout(
            'Train Epoch: ' +
            '{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                ii,
                len(self.train_set),
                100. * ii / len(self.train_set),
                epoch_loss / ii))

    def eval(self, epoch):
        self.vae.eval()
        epoch_loss = 0.
        with torch.no_grad():
            for ii in poem(
                    range(len(self.val_set)), "EVAL EPOCH " + str(epoch)):
                X, t, _ = self.val_set[ii]
                X, t = X.to(self.device), t.to(self.device)
                y, _, _ = self.vae(X)

                loss = F.binary_cross_entropy(
                    (1.0 + y.reshape(X.shape[0], -1)) / 2,
                    (1.0 + t.reshape(X.shape[0], -1)) / 2,
                    reduction='sum')

                epoch_loss += loss.item()

        Tools.pyout(
            'Eval Epoch: ' +
            '{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                ii,
                len(self.val_set),
                100. * ii / len(self.val_set),
                epoch_loss / ii))
        if loss < self.best_loss:
            self.best_loss = loss
            self.vae.save_state_dict(self.save_loc)
