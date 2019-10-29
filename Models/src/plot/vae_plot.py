import cv2
import json
import os
import numpy as np
import torch


from src.alignment.align_matrix import AlignMatrix
from src.modules.vae import VAE
from src.plot.cv_hist import CVHist
from src.utils.tools import Tools
poem = Tools.tqdm_pbar


class VAEPlot:
    def __init__(self,
                 device=None,
                 state_dict_root=None,
                 root=None):
        self.device = device
        self.POS = tuple([Tools.fname(f)
                          for f in Tools.list_dirs(Tools.list_dirs(root)[0])])
        self.VAE_DICT, self.EMB_SIZE = self._load_vae_dicts(root)
        am = AlignMatrix(root)
        self.alignments = am.load()
        self.VAE = [VAE(state_dict_path=os.path.join(
                    state_dict_root, pos, 'vae_mdl.pth')).to(device)
                    for pos in poem(self.POS, "LOADING VAE MODELS")]
        self.cv_hist, self.labels = self._init_hist(
            (240, 240 * len(self.POS)), root)
        self.lbl_dict = self._init_labels(root)

    def visualize(self, in_folder, ou_folder):
        Tools.makedirs(ou_folder)

        self.len = [len(Tools.list_files(os.path.join(in_folder, pos)))
                    for pos in self.POS]

        frames = [Tools.fname(f) for f in Tools.list_files(
            os.path.join(in_folder, min(self.POS, key=lambda x: len(x))))]

        n_zfill = len(frames[0].split('.')[0])

        # define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(os.path.join(
            ou_folder, in_folder.split('/')[-1] + '.mp4'),
            fourcc, 16,
            (240 * len(self.POS),
             480 if self.lbl_dict is None else 720))

        # init frame
        # if os.path.exists(in_folder.replace('_view', '') + '.json'):
        if self.lbl_dict is not None:
            main_frm = np.zeros((720, len(self.POS) * 240, 3), dtype=np.uint8)
        else:
            main_frm = np.zeros((480, len(self.POS) * 240, 3), dtype=np.uint8)

        for fii, frame in poem(enumerate(frames), Tools.fname(in_folder),
                               total=len(frames)):
            try:
                vae_embds = self._imagine(in_folder, fii, n_zfill)
                for pii, pos in enumerate(self.POS):
                    orig_img = cv2.imread(os.path.join(in_folder, pos, frame))
                    orig_img = cv2.resize(orig_img, (240, 240))
                    main_frm[0:240, pii * 240:(pii + 1) * 240, :] = \
                        np.copy(orig_img[:, :, :3])

                    with torch.no_grad():
                        X = torch.FloatTensor(vae_embds[pos]).to(
                            self.device).unsqueeze(0)
                        y = self.VAE[pii].decode(X)
                        imag_img = y.cpu().numpy().squeeze()
                    imag_img = (imag_img + 1.) * 127.5
                    imag_img = imag_img.transpose(
                        1, 2, 0)[:, :, ::-1].astype(np.uint8)
                    imag_img = cv2.resize(imag_img, (240, 240))
                    main_frm[240:480, pii * 240:(pii + 1) * 240, :] = \
                        np.copy(imag_img)
                if self.lbl_dict is not None:
                    est_lbls = self._estimate_labels(in_folder, fii, n_zfill)
                    vals = tuple(
                        [(float(self.lbl_dict[in_folder][frame][lbl]), el)
                         for lbl, el in zip(self.labels, est_lbls)])
                    main_frm[480:720, :, :] = self.cv_hist.plot(values=vals)
            except IndexError:
                pass
            writer.write(main_frm)
        writer.release()

    def _estimate_labels(self, in_trial, frame_idx, n_zfill):
        vals = []
        denom = []

        for anchor_trial in [f for f in self.alignments[in_trial]
                             if 'fake' not in f]:
            match_frm, weight = \
                self.alignments[in_trial][anchor_trial][frame_idx]
            vals.append([float(self.lbl_dict[anchor_trial][match_frm][lbl])
                         for lbl in self.labels])
            denom.append(weight)
        vals = np.array(vals)
        denom = np.array(denom, dtype=np.float32)
        denom /= np.sum(denom)
        vals = np.average(vals, axis=0, weights=denom)
        return tuple(vals)

    def _imagine(self, in_trial, frame_idx, n_zfill):
        embeddings = {pos: [] for pos in self.POS}
        denom = []

        for anchor_trial in [f for f in self.alignments[in_trial]
                             if 'fake' not in f]:
            match_frm, weight = \
                self.alignments[in_trial][anchor_trial][frame_idx]

            for pos in self.POS:
                embeddings[pos].append(
                    self.VAE_DICT[anchor_trial]
                    [pos][match_frm])
            denom.append(weight)

        denom = np.array(denom, dtype=np.float32)
        denom = np.exp(denom)
        denom /= np.sum(denom)

        for pos in self.POS:
            embeddings[pos] = np.array(embeddings[pos], dtype=np.float32)
            embeddings[pos] = np.average(
                embeddings[pos], axis=0)

        return embeddings

    def _load_vae_dicts(self, root):
        vae_dict = {}
        emb_size = None
        for trial_folder in poem(
                Tools.list_dirs(root), "LOADING VAE EMBEDDINGS"):
            vae_dict[trial_folder] = {}
            for pos in self.POS:
                vae_dict[trial_folder][pos] = {}
                with open(os.path.join(
                        trial_folder, pos + '_vae.json'), 'r') as f:
                    data = json.load(f)
                for frm in list(data):
                    vae_dict[trial_folder][pos][frm] = data[frm]
                    if emb_size is None:
                        emb_size = len(data[frm])
        return vae_dict, emb_size

    def _init_labels(self, root):
        lbl_dict = {}
        for label_file in Tools.list_files(root, end='.json'):
            with open(label_file, 'r') as f:
                if os.path.exists(label_file.replace('.json', '_view')):
                    lbl_dict[label_file.replace('.json', '_view')] = \
                        json.load(f)
                else:
                    lbl_dict[label_file.replace('.json', '')] = json.load(f)
        if len(lbl_dict) == 0:
            return None
        else:
            return lbl_dict

    def _init_hist(self, shape, root):
        labels = []
        minvals = []
        maxvals = []
        for label_file in Tools.list_files(root, end='.json'):
            # Tools.pyout(label_file)
            with open(label_file, 'r') as f:
                d = json.load(f)
            for key in d:
                # Tools.pyout(key, d[key]['contact'])
                for lbl in d[key]:
                    if lbl not in labels:
                        labels.append(lbl)
                        minvals.append(float(d[key][lbl]))
                        maxvals.append(float(d[key][lbl]))
                    else:
                        minvals[labels.index(lbl)] = min(
                            minvals[labels.index(lbl)], float(d[key][lbl]))
                        maxvals[labels.index(lbl)] = max(
                            maxvals[labels.index(lbl)], float(d[key][lbl]))
        if len(labels) > 0:
            return CVHist(shape, labels, minvals, maxvals, n_bars=2), labels
        else:
            return None, None
