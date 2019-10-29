import torch
import torch.optim as optim

from src.modules.tcn_vgg import TCN
from src.datasets.pil_set import PILSet
from src.utils.tools import Tools
from src.utils.loss import Loss
from src.utils.config import Config
poem = Tools.tqdm_pbar


class TCNTrainer:
    def __init__(self,
                 devices=(None, None),
                 state_dict_paths=(None, None),
                 train_root=None,
                 val_root=None,
                 save_loc=None):
        self.devices = devices
        self.state_dict_paths = state_dict_paths
        self.save_loc = save_loc

        self.train_set = PILSet(
            root_dir=train_root,
            augment=True)
        self.val_set = PILSet(
            root_dir=val_root,
            augment=False)

        self.tcn = TCN(devices=devices, state_dict_paths=state_dict_paths)
        self.tcn.switch_mode('train')
        self.optimizer = optim.Adam(list(filter(lambda p: p.requires_grad,
                                                self.tcn.parameters())))

    def train_loop(self, epochs_start, epochs_end):
        self.best_ratio = 1.
        self.last_improvement = 0
        for epoch in poem(range(epochs_start, epochs_end), "TRAINING TCN"):
            self.train(epoch)
            self.test(epoch)
            if epoch - 10 > self.last_improvement:
                break
        self.train_set.close()
        self.val_set.close()

    def train(self, epoch):
        self.tcn.switch_mode('train')

        epoch_loss = 0.
        cum_loss = 0.
        cum_ratio_loss = 0.
        n = 0
        for idx in poem(range(len(self.train_set)), "train " + str(epoch)):
            X, labels, perspectives, paths = self.train_set[idx]
            if X is None:
                continue
            n += 1
            X = X.to(self.devices[0])
            labels = labels.to(self.devices[1])
            perspectives = perspectives.to(self.devices[1])

            self.optimizer.zero_grad()
            y = self.tcn(X)

            assert not Tools.contains_nan(y)

            loss = Loss.triplet_semihard_loss(
                y, labels, perspectives,
                margin=Config.TCN_MARGIN, device=self.devices[1])
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            cum_loss += Loss.embedding_accuracy(
                y, labels, perspectives, device=self.devices[1]).item()
            cum_ratio_loss += Loss.embedding_accuracy_ratio(
                y, labels, perspectives)

        Tools.pyout(
            'Train Epoch: ' +
            '{} [{}/{} ({:.0f}%)]\tAccuracy: '
            '{:.6f}\tRatio: {:.6f}\tLoss: {:.6f}'.format(
                epoch,
                idx,
                len(self.train_set),
                100. * idx / len(self.train_set),
                cum_loss / n,
                cum_ratio_loss / n,
                epoch_loss / (n)))
        Tools.log(
            'Train Epoch: ' +
            '{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                n,
                len(self.train_set),
                100.,
                epoch_loss / (n)))

    def test(self, epoch):
        self.tcn.switch_mode('eval')
        cum_loss = 0.
        cum_ratio_loss = 0.
        n = 0
        for idx in poem(range(len(self.val_set)), "eval " + str(epoch)):
            # if idx == 5:
            #     break
            with torch.no_grad():
                batch = self.val_set[idx]
                X, labels, perspectives = batch[0], batch[1], batch[2]

                if X is None:
                    continue
                n += 1

                X = X.to(self.devices[0])
                labels = labels.to(self.devices[1])
                perspectives = perspectives.to(self.devices[1])

                y = self.tcn(X)

                assert not Tools.contains_nan(y)

                cum_loss += Loss.embedding_accuracy(
                    y, labels, perspectives, device=self.devices[1]).item()
                cum_ratio_loss += Loss.embedding_accuracy_ratio(
                    y, labels, perspectives)

        Tools.pyout(
            'Test Epoch: ' +
            ('{} [{}/{} ({:.0f}%)]\tAccuracy: '
                '{:.6f}\tRatio: {:.6f}').format(
                    epoch,
                    n,
                    len(self.val_set),
                    100. * idx / len(self.val_set),
                    cum_loss / (n),
                    cum_ratio_loss / (n)))
        Tools.log(
            'Test Epoch: ' +
            ('{} [{}/{} ({:.0f}%)]\tAccuracy: '
                '{:.6f}\tRatio: {:.6f}').format(
                    epoch,
                    n,
                    len(self.val_set),
                    100.,
                    cum_loss / (n),
                    cum_ratio_loss / (n)))
        if cum_ratio_loss / (n) < self.best_ratio:
            self.best_ratio = cum_ratio_loss / (n)
            self.save_state_dict(self.save_loc)
            self.last_improvement = epoch

    def val_with_labels(self, epoch):
        """inference model on validation set, for each frame in each video
        find nearest neighbor in each other video and check accuracy on
        labels of that frame
        """
        self.tcn.switch_mode('eval')
        Y = []  # tcn embeddings
        V = []  # video names
        R = []  # video classes
        L = []  # frame features

        # compute embeddings for all frames
        with torch.no_grad():
            for batch_idx in range(len(self.val_set)):
                batch = self.val_set[batch_idx]
                X, v, r, lbls = batch[0], batch[1], batch[2], batch[3]
                X = X.to(self.devices[0])
                y = self.tcn(X).cpu().numpy()
                Y.extend(list(y))
                V.extend(v)
                R.extend(r)
                L.extend(lbls)
                print("Validation Epoch {}: {}/{} ({:.0f}%)".format(
                    epoch,
                    batch_idx,
                    len(self.val_set),
                    100. * batch_idx / len(self.val_set)), '    ', end='\r')

        accuracy, correct, N = Loss.labeled_accuracy(Y, V, R, L)

        Tools.log(
            'Validation Epoch {}: Accuracy {:.0f}/{} ({:.0f}%)'.format(
                epoch,
                correct,
                N,
                100. * accuracy))

    def save_state_dict(self, name):
        self.tcn.save_state_dicts(name)
