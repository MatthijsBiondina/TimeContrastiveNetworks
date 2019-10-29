import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys

from src.utils.tools import Tools


class Aligner:
    def __init__(self):
        pass

    def make_align_dict(self, root, trial_root, lock):
        dict_ = {}
        self.root = root
        self.trial_root = trial_root
        for anchor_root in [f for f in Tools.list_dirs(root)
                            if 'fake' not in f]:
            self.anchor_root = anchor_root
            self.trial_root = trial_root
            if not trial_root == anchor_root:
                dict_[anchor_root] = self._align(trial_root, anchor_root, lock)

        with open(os.path.join(trial_root, 'alignment.json'), 'w+') as f:
            json.dump(dict_, f, indent=1)

    def _align(self, trial_root, anchor_root, lock):
        """
        project trial sequence onto anchor sequence with modified dtw
        """
        try:

            frm_a, emb_a = self._init_embeddings(anchor_root, lock)
            frm_t, emb_t = self._init_embeddings(trial_root, lock)

            dist_matrix = self._compute_dist_matrix(emb_t, emb_a)
            cost_matrix = self._compute_cost_matrix(dist_matrix)
            idx_path = self._compute_path(cost_matrix)

            path = []
            for point in idx_path:
                path.append(
                    (frm_a[point[1]],
                        1e-3 / dist_matrix[point[1], point[0]]))

            # self._plot(path)

            return path
        except Exception as e:
            Tools.debug(e, ex=1)
            sys.exit(1)

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
        Y = distances.shape[0]
        dy = 10

        for yii in range(distances.shape[0]):
            accumulated_cost[yii, 0] = distances[yii, 0]
        for xii in range(1, distances.shape[1]):
            for yii in range(distances.shape[0]):

                try:
                    accumulated_cost[yii, xii] =  \
                        distances[yii, xii] + \
                        np.min(accumulated_cost[max(
                            0, yii - dy):min(Y, yii + dy), xii - 1])
                except Exception as e:
                    Tools.debug(distances.shape,
                                max(0, yii - 3),
                                min(yii + 3,
                                    distances.shape[0]), xii - 1)
                    raise e
            # Tools.render(self._heatmap(accumulated_cost), waitkey=10)
        return accumulated_cost

    def _compute_path(self, cost_matrix):
        Y = cost_matrix.shape[0]
        dy_plus = 20
        dy_min = 5

        path = [(0, 0)]

        for xii in range(1, cost_matrix.shape[1]):
            # for xii in range(cost_matrix.shape[1] - 2, -1, -1):
            path.append((
                xii,
                min(
                    range(max(0, path[-1][1] - dy_min),
                          min(Y, path[-1][1] + dy_plus)),
                    # range(path[-1][1] + 1),  #indices are legal
                    key=lambda y: cost_matrix[y, xii])))
        return path

    def _plot(self, path):
        appendage = '/steady' if 'pouring' in self.trial_root else '/left'
        frames = Tools.list_files(self.trial_root + appendage)
        a_fldr = self.anchor_root + appendage

        for ii in range(len(path)):
            frame = cv2.imread(frames[ii])
            ancho = cv2.imread(os.path.join(a_fldr, path[ii][0]))
            # Tools.debug(frame.shape, ancho.shape)
            Tools.render(np.concatenate((frame, ancho), axis=1))

    def _heatmap(self, matrix, path=None):
        """
        draw a heatmap of distance matrix or accumulated cost matrix with
        selected path
        """

        img = np.copy(matrix)
        img -= np.min(img)
        img /= np.max(img)

        if path is None:
            img = cv2.resize(img, (360, 360))

            return img[::-1, :]
        else:
            img = (np.stack((img,) * 3, axis=-1) * 255).astype(np.uint8)
            appendage = '/steady' if 'pouring' in self.trial_root else '/left'
            frames = Tools.list_files(self.trial_root + appendage)
            anchors = Tools.list_files(self.anchor_root + appendage)
            # print(len(frames), len(path))
            for ii in range(len(path) - 2, -1, -1):
                p0 = path[ii]
                p1 = path[ii + 1]

                img = cv2.line(img, p0, p1, (0, 0, 255), 2)
                frame = cv2.imread(frames[p0[0]])
                ancho = cv2.imread(anchors[p0[1]])

                frame = cv2.resize(frame, (240, 240))
                ancho = cv2.resize(ancho, (240, 240))

                if not ii % 3:
                    img_ = np.copy(img)
                    img_ = cv2.resize(img_, (240, 240))
                    Tools.render(np.concatenate(
                        (img_[::-1, :, :], frame[:, :, ::1], ancho), axis=1),
                        waitkey=50)
            return img[::-1, :, :]

            # Tools.render(img, waitkey=50)

            # fig, ax = plt.subplots()
            # plt.imshow(matrix, interpolation='nearest', cmap='terrain')
            # plt.gca().invert_yaxis()
            # plt.xlabel("X")
            # plt.ylabel("Y")
            # plt.grid()
            # plt.colorbar()
            # if path is not None:
            #     plt.plot([point[0] for point in path],
            #              [point[1] for point in path], 'r')
            # fig.canvas.draw()
            # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            # data = data.reshape(fig.canvas.get_width_height()
            #                     [::-1] + (3,))[:, :, ::-1]
            # plt.close()
            # Tools.render(data)
