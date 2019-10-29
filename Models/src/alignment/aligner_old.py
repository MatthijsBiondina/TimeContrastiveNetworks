import json
import matplotlib.pyplot as plt
import numpy as np
import os

from src.utils.tools import Tools


class Aligner:
    def __init__(self):
        pass

    def make_align_dict(self, root, trial_root, lock):
        dict_ = {}
        for anchor_root in Tools.list_dirs(root):
            if not trial_root == anchor_root:
                dict_[anchor_root] = self._align(trial_root, anchor_root, lock)

        with open(os.path.join(trial_root, 'alignment.json'), 'w+') as f:
            json.dump(dict_, f, indent=1)

    def _align(self, trial_root, anchor_root, lock):
        """
        project trial sequence onto anchor sequence with modified dtw
        """
        frm_a, emb_a = self._init_embeddings(anchor_root, lock)
        frm_t, emb_t = self._init_embeddings(trial_root, lock)

        dist_matrix = self._compute_dist_matrix(emb_t, emb_a)
        cost_matrix = self._compute_cost_matrix(dist_matrix)
        idx_path = self._compute_path(cost_matrix)

        path = []
        for point in idx_path[::-1]:
            path.append(
                (frm_a[point[1]],
                 1e-3 / dist_matrix[point[1], point[0]]))
        return path

    def _init_embeddings(self, root, lock):
        embd_files = Tools.list_files(root, end='.json', substr='embed_')
        # lock.acquire()
        with open(embd_files[0], 'r') as f:
            sample_dict = json.load(f)
            frames = list(sample_dict)
            nr_frames = len(sample_dict)
            emb_N = len(sample_dict[list(sample_dict)[0]])
            embeddings = np.zeros(
                (nr_frames, emb_N * len(embd_files)), dtype=float)
        # lock.release()

        for ii, emb_F in enumerate(embd_files):
            # lock.acquire()
            with open(emb_F, 'r') as f:
                dict_ = json.load(f)
            # lock.release()
            for jj, frame in enumerate(sorted(list(dict_))):
                embeddings[jj, ii * emb_N:(ii + 1) * emb_N] = \
                    np.array(dict_[frame])

        return frames, embeddings

    def _compute_dist_matrix(self, X, Y):
        distances = np.zeros((Y.shape[0], X.shape[0]))

        for yii in range(Y.shape[0]):
            for xii in range(X.shape[0]):
                distances[yii, xii] = np.mean((X[xii, :] - Y[yii, :])**2)

        return distances

    def _compute_cost_matrix(self, distances):
        accumulated_cost = np.zeros(distances.shape)
        for yii in range(distances.shape[0]):
            accumulated_cost[yii, 0] = distances[yii, 0]
        for xii in range(1, distances.shape[1]):
            accumulated_cost[0, xii] = distances[0, xii] + \
                accumulated_cost[0, xii - 1]
        for yii in range(1, distances.shape[0]):
            for xii in range(1, distances.shape[1]):
                accumulated_cost[yii, xii] = \
                    np.min(accumulated_cost[:yii + 1, xii - 1]
                           ) + distances[yii, xii]
        return accumulated_cost

    def _compute_path(self, cost_matrix):
        path = [(
            cost_matrix.shape[1] - 1,
            min(range(cost_matrix.shape[0]),
                key=lambda y: cost_matrix[y, -1]))]
        for xii in range(cost_matrix.shape[1] - 2, -1, -1):
            path.append((
                xii,
                min(range(path[-1][1] + 1),
                    key=lambda y: cost_matrix[y, xii])))
        return path

    def _heatmap(self, matrix, path=None):
        """
        draw a heatmap of distance matrix or accumulated cost matrix with
        selected path
        """
        fig, ax = plt.subplots()
        plt.imshow(matrix, interpolation='nearest', cmap='terrain')
        plt.gca().invert_yaxis()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.colorbar()
        if path is not None:
            plt.plot([point[0] for point in path],
                     [point[1] for point in path], 'r')
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()
                            [::-1] + (3,))[:, :, ::-1]
        plt.close()
        Tools.render(data)
