import cv2
import joblib
import json
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering
import sys

from src.plot.cv_plot import CVPlot
from src.utils.tools import Tools
poem = Tools.tqdm_pbar


class WTCluster:
    def __init__(self, joblib_path=None, n_clusters=5, averaged=False):
        self.reduction = joblib.load(joblib_path)
        self.n_clusters = n_clusters
        self.averaged = averaged

    def visualize(self, in_folder, ou_folder):
        Tools.makedirs(ou_folder)
        data = self._load_embedding_dicts(in_folder)
        main_frm = np.zeros((480, len(data) * 240, 3), dtype=np.uint8)
        N = data[0][2].shape[0]

        # define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(os.path.join(
            ou_folder, in_folder.split('/')[-1] + '.mp4'),
            fourcc, 16, (240 * len(data), 480))
        Tools.pyout(os.path.join(
            ou_folder, in_folder.split('/')[-1] + '.mp4'))

        for ii, pos_data in enumerate(data):
            data[ii] = self._cluster(*pos_data)

        for n in poem(range(N), Tools.fname(in_folder)):
            for ii, pos_data in enumerate(data):
                fldr, frms, embd, rdcd, lbls, plot = pos_data

                # load frame image
                img_pth = os.path.join(fldr, frms[n])
                frame = cv2.imread(img_pth)
                frame = cv2.resize(frame, (240, 240)).astype(np.uint8)

                # draw cluster color in frame
                color = plot._prog2color(lbls[n])
                cv2.rectangle(
                    frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 10)

                # draw embeddings
                plt_img = plot.plot(rdcd[n], color=color, scope=False)

                # add subfigures to frame
                main_frm[0:240, ii * 240:(ii + 1) * 240, :] = frame[:, :, :]
                main_frm[240:480, ii * 240:(ii + 1) * 240, :] = \
                    plt_img[:, :, :]
            # Tools.render(main_frm, waitkey=50, name="SEGMENTATION")
            writer.write(main_frm)
        writer.release()
        # cv2.destroyAllWindows()

    def _cluster(self, fldr, frms, embd, rdcd):
        connectivity_matrix = self._connectivity_matrix(
            embd.shape[0])
        model = AgglomerativeClustering(linkage='ward',
                                        connectivity=connectivity_matrix,
                                        n_clusters=self.n_clusters)
        model.fit(embd)
        lbls = self._reorder(model.labels_)
        plot = CVPlot()
        return (fldr, frms, embd, rdcd, lbls, plot)

    def _reorder(self, labels):
        labels_ = np.zeros((len(labels),), dtype=float)
        current = labels[0]
        cluster_idx = 0
        for ii, label in enumerate(labels):
            if not label == current:
                current = label
                cluster_idx += 1
            labels_[ii] = cluster_idx
        return labels_ / np.max(labels_)

    def _connectivity_matrix(self, N):
        mtrx = np.zeros((N, N), dtype=float)
        np.fill_diagonal(mtrx[1:, :], 1)
        np.fill_diagonal(mtrx[:, 1:], 1)
        return mtrx

    def _reduce(self, data):
        data_reduced = self.reduction.transform(data)
        data_reduced /= max(abs(np.min(data_reduced)),
                            abs(np.max(data_reduced)))
        return data_reduced

    def _load_embedding_dicts(self, folder):
        try:
            out = []
            for json_filename in sorted(Tools.list_files(
                    folder, end='.json', substr='embed_')):
                pos = Tools.fname(json_filename).replace(
                    'embed_', '').replace('.json', '')
                root_folder = os.path.join(folder, pos)

                with open(json_filename, 'r') as f:
                    D = json.load(f)
                X = np.zeros((len(D), len(D[list(D)[0]])), dtype=float)
                frames = sorted(list(D))
                for ii, frame in enumerate(frames):
                    X[ii, :] = np.array(D[frame])

                out.append((root_folder, frames, X, self._reduce(X)))

            if self.averaged:
                X_avg = np.stack([pos_data[2] for pos_data in out], axis=0)
                X_avg = np.mean(X_avg, axis=0)
                for ii, (fldr, frms, _, rdcd) in enumerate(out):
                    out[ii] = (fldr, frms, X_avg, rdcd)
            return out
        except Exception as e:
            Tools.debug(e)
            Tools.debug(folder, ex=0)
