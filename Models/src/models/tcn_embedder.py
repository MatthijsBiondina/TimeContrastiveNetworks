import json
import numpy as np
import os
import sys
# import time
import torch
from torch import autograd
from tqdm import tqdm

from src.datasets.embedder_set import EmbedderSet
from src.modules.tcn_vgg import TCN
from src.utils.tools import Tools
pyout = Tools.pyout
poem = Tools.tqdm_pbar


class TCNEmbedder:
    def __init__(self,
                 devices=(None, None),
                 state_dict_paths=(None, None),
                 roots=(None,)):
        self.devices = devices
        self.state_dict_paths = state_dict_paths
        self.roots = roots

        self.tcn = TCN(devices=self.devices,
                       state_dict_paths=self.state_dict_paths)

    def embed(self, force=False):
        self.tcn.switch_mode('eval')
        for root in poem(sorted(self.roots, reverse=True), "EMBEDDING"):
            for trial in poem(Tools.list_dirs(root), Tools.fname(root)):
                for pos_path in poem(Tools.list_dirs(trial),
                                     Tools.fname(trial)):
                    pos = pos_path.split('/')[-1]
                    dataset = EmbedderSet(
                        root_dir=pos_path)
                    embeddings = {}
                    for X, paths in poem(dataset, Tools.fname(pos_path)):
                        if len(paths) > 0:
                            y = self._fwd(X)
                            if not np.isfinite(y).all():
                                pyout('before', paths)
                                sys.exit(0)

                            for ii, path in enumerate(paths):
                                embeddings[path.split(
                                    '/')[-1]] = np.copy(y[ii, :]).tolist()
                    for key in embeddings:
                        if np.isnan(embeddings[key]).any():
                            pyout('after', pos_path, key)
                            sys.exit(0)

                    with open(os.path.join(
                            trial, 'embed_' + pos + '.json'), 'w+') as f:
                        json.dump(embeddings, f, indent=1)
                    del dataset
                    del embeddings

    def _contains_nans(self, path):
        try:
            with open(path, 'r') as f:
                embeddings = json.load(f)
        except FileNotFoundError:
            return True
        for key in embeddings:
            if np.isnan(embeddings[key]).any():
                return True
        return False

    def _fwd(self, X):
        with torch.no_grad():
            with autograd.detect_anomaly():
                try:
                    X = X.to(self.devices[0])
                    y = self.tcn(X).detach().cpu().numpy()
                except Exception as e:
                    os.system('clear')
                    Tools.pyout(e, force=True)
                    sys.exit(0)
        return y
