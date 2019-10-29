import csv
from fastdtw import fastdtw
import json
import multiprocessing
import numpy as np
import os
from scipy.spatial.distance import euclidean
from sklearn.cluster import AgglomerativeClustering

import sklearn.manifold as manifold
import warnings

from src.utils.config import Config
from src.utils.tools import Tools


class ScatterPlot:
    def __init__(self,
                 joblib_path=None,
                 data_root=None,
                 n_components=2,
                 n_clusters=1):
        self._init_model(data_root, n_components, n_clusters)

    def _init_model(self, data_root, n_components, n_clusters):

        # Load embeddings
        trials_embeddings = []
        for trial_folder in Tools.tqdm_pbar(
                Tools.list_dirs(data_root), 'LOAD TCN EMBEDDINGS'):
            trials_embeddings.append(self._load_embeddings(trial_folder))

        # compute distance matrix & perform dimensionality reduction
        dist_matrix = self._compute_dist_matrix(trials_embeddings)
        reduce_embd = self._reduce(dist_matrix, n_components=n_components)

        # do clustering
        labels = self._cluster(dist_matrix, n_clusters=n_clusters)

        # write results to csv
        self._write_csv(data_root, trials_embeddings, reduce_embd, labels)

    def _load_embeddings(self, trial_folder):
        X = []
        for embed_json in Tools.search(trial_folder, ('embed_',), substr=True):
            with open(embed_json, 'r') as f:
                D = json.load(f)
            x = np.zeros((len(list(D)), Config.TCN_EMB_SIZE), dtype=float)
            for ii, key in enumerate(sorted(list(D))):
                x[ii, :] = np.array(D[key])
            X.append(np.copy(x))
        X = np.stack(X, axis=0)
        X = np.mean(X, axis=0)
        return (trial_folder, X)

    def _compute_dist_matrix(self, embeddings):
        N = len(embeddings)
        dist_matrix = np.zeros((N,) * 2, dtype=float)

        tasks = []
        for ii in range(len(embeddings)):
            for jj in range(ii + 1, len(embeddings)):
                tasks.append(
                    (embeddings[ii][1], embeddings[jj][1], ii, jj))

        pool = multiprocessing.Pool(15)
        try:
            for retval in Tools.tqdm_pbar(
                    pool.imap_unordered(self._dtw, tasks),
                    description='DISTANCE MATRIX', total=len(tasks)):
                ii, jj = retval[0]
                dist_matrix[ii, jj], dist_matrix[jj, ii] = (retval[1],) * 2
        finally:
            pool.close()
            pool.join()

        return dist_matrix

    def _reduce(self, dist_matrix, n_components=2):
        Tools.pyout("FITTING UMAP...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = manifold.MDS(n_components=n_components,
                                 metric='precomputed')
            reduced = np.array(model.fit_transform(dist_matrix))
        Tools.pyout("---------> DONE")
        reduced -= np.min(reduced, axis=0)
        reduced /= np.max(reduced, axis=0)
        reduced = (reduced - 0.5) * 2
        return reduced.tolist()

    def _cluster(self, dist_matrix, n_clusters=15):
        Tools.pyout("CLUSTERING")
        model = AgglomerativeClustering(affinity='precomputed',
                                        linkage='average',
                                        n_clusters=n_clusters)
        model.fit(dist_matrix)
        colors_ = []
        for lbl in model.labels_:
            colors_.append(Tools.num2hslstr(lbl / max(model.labels_)))
        Tools.pyout("----> DONE")
        return colors_

    def _write_csv(self, data_root, trials_embeddings, reduce_embd, labels):
        Tools.pyout("WRITING CSV FILE")
        SAVE_ROOT = '/media/roblaundry/' + \
            data_root.split('/')[-2] + '/results/BT'
        Tools.makedirs(SAVE_ROOT)
        with open(os.path.join(SAVE_ROOT, 'data.csv'), 'w+') as csv_file:
            fieldnames = ['trial_name', 'x', 'y', 'c']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            for ii, embedding in enumerate(trials_embeddings):
                writer.writerow({
                    'trial_name': Tools.fname(embedding[0]),
                    'x': reduce_embd[ii][0],
                    'y': reduce_embd[ii][1],
                    'c': labels[ii].replace(',', ';')})
        Tools.pyout("----------> DONE")

    def _dtw(self, args):
        embd_ii, embd_jj, ii, jj = args
        dtw, _ = fastdtw(embd_ii, embd_jj, dist=euclidean)
        dtw /= max(embd_ii.shape[0], embd_jj.shape[0])
        return ((ii, jj), dtw)
