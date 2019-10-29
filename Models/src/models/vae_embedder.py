import json
import os
import torch

from src.datasets.vae_set import VAESet
from src.modules.vae import VAE
from src.utils.tools import Tools
poem = Tools.tqdm_pbar


class VAEEmbedder:
    def __init__(self,
                 device=None,
                 state_dict_path=None,
                 root=None,
                 pos=None):
        self.device = device
        self.state_dict_path = state_dict_path
        self.root = root
        assert pos is not None
        self.pos = pos

        self.dataset = VAESet(
            root_dir=root,
            pos=pos,
            augment=False)

        self.vae = VAE(state_dict_path=state_dict_path).to(self.device)
        self.vae.eval()

    def embed(self):
        dict_ = None
        folder = None

        with torch.no_grad():
            for ii in poem(range(len(self.dataset)),
                           "STORING EMBEDDINGS " + self.pos):
                X, _, paths = self.dataset[ii]
                X = X.to(self.device)
                _, mu, _ = self.vae(X)
                mu = mu.cpu().numpy().tolist()

                for pth, emb in zip(paths, mu):
                    if not folder == os.path.join(*pth.split('/')[:-1]):
                        if dict_ is not None:
                            with open('/' + folder + '_vae.json', 'w+') as f:
                                json.dump(dict_, f, indent=1)
                        folder = os.path.join(*pth.split('/')[:-1])
                        dict_ = {}
                    dict_[Tools.fname(pth)] = emb

        with open('/' + folder + '_vae.json', 'w+') as f:
            json.dump(dict_, f, indent=1)

            # Tools.debug(folder, Tools.fname(pth))

            # Tools.debug(paths)
            # Tools.debug(1, ex=0)
