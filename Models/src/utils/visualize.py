import torch
import numpy as np
import os
from copy import deepcopy
from sklearn.decomposition import PCA
import sys
import shutil

import cv2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import Axes3D

from src.utils.tools import Tools
from src.utils.preprocessing import Prep


class Visualizer:
    def __init__(self,
                 model=None,
                 dataset=None):
        if model and dataset:
            fig = plt.figure()
            self.ax = fig.gca(projection='3d')
            # self.ax = fig.add_subplot(111, aspect='equal')
            self.model = model
            self.dataset = dataset
            self._init_pca(self.model, self.dataset)

    def alpha_video(self, folder):
        D = Prep.init_joint_dict(os.path.join(
            folder, '3d', 'alphapose-results.json'))

        for img_pth in sorted(Tools.list_files(folder)):
            img_name = img_pth.split('/')[-1]
            img = cv2.imread(img_pth)
            wait = 1
            try:
                person = D[img_name][0]
                lwrist = (int(person['KP']['LWrist'][0]),
                          int(person['KP']['LWrist'][1]))
                rwrist = (int(person['KP']['RWrist'][0]),
                          int(person['KP']['RWrist'][1]))
                col = (0, 0, 255)
                if lwrist[0] < 0 or lwrist[1] < 0 or rwrist[0] < 0 or lwrist[1] < 0:
                    print('foo')
                    wait = 5000
                    col = (0, 255, 0)
                cv2.circle(img, lwrist, 10, col, thickness=-1)
                # cv2.circle(img, rwrist, 10, (0, 0, 255), thickness=-1)
                # print(person['KP']['RWrist'], person['KP']['LWrist'])
            except KeyError:
                pass

            cv2.imshow('vid', img)
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                sys.exit(0)

    def _init_pca(self, model, dataset):
        self.model.switch_mode('eval')
        self.Y = []
        self.V = []
        with torch.no_grad():
            for batch_idx in range(len(dataset)):
                print('TCN-ing',
                      int(batch_idx / len(dataset) * 100), '%')

                x, v = dataset[batch_idx]
                x = x.to(self.model.devices[0])
                y = self.model(x).cpu().numpy()
                self.Y.extend(list(y))
                self.V.extend(v)

        self.Y = np.array(self.Y)
        self.pca = self._pca_fit(self.Y, n_components=3)

    def _pca_fit(self, data, n_components=3):
        X = deepcopy(data)
        pca = PCA(n_components=n_components)
        pca.fit(X)
        return pca

    def visualize(self):
        traj = []
        v_name = '/'.join(self.V[0].split('/')[:-1])

        for n, (v, y) in enumerate(zip(self.V, self.Y)):
            print('VISUALIZING:', int(n / len(self.V) * 100), '%')
            plot_pth = v.replace('datasets', 'plots')
            fold_pth = '/'.join(plot_pth.split('/')[:-1])
            try:
                os.makedirs(fold_pth)
            except OSError:
                pass

            if not '/'.join(v.split('/')[:-1]) == v_name:
                v_name = '/'.join(v.split('/')[:-1])
                traj = []

            traj.append(self.pca.transform(np.array([y])).squeeze())
            # print(traj)
            self._plot(np.array(traj))
            plt.savefig(plot_pth)
        self._paste()

    def _plot(self, traj):
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.plot([-1, 1], [-1, 1], [-1, 1], alpha=0.)
        for i in range(0, 20):
            self.ax.plot(
                [1., 1.],
                [-1. + i * 0.1, -1. + (i + 1) * 0.1],
                [-1., -1.],
                '-',
                color=plt.cm.gnuplot(
                    max(0., min(1., (-1. + (i + 1) * 0.1 + 1) / 2))))

        for i in range(traj.shape[0] - 1):
            c = plt.cm.gnuplot(max(0., min(1., (traj[i + 1, 1] + 1) / 2)))
            self.ax.plot(
                traj[i:i + 2, 0],
                traj[i:i + 2, 1],
                traj[i:i + 2, 2],
                '-',
                color=(c[0], c[1], c[2], i / traj.shape[0]))
        self.ax.scatter(
            traj[-1, 0],
            traj[-1, 1],
            traj[-1, 2],
            'o',
            color=plt.cm.gnuplot(max(0., min(1., (traj[-1, 1] + 1) / 2))))

    def _paste(self):
        root_dir = self.dataset.root.replace('datasets', 'plots')
        for trial_pth in Tools.list_dirs(root_dir):
            l_pths = sorted(Tools.list_files(os.path.join(trial_pth, '0')))
            m_pths = sorted(Tools.list_files(os.path.join(trial_pth, '1')))
            r_pths = sorted(Tools.list_files(os.path.join(trial_pth, '2')))

            minlen = min(len(l_pths), len(m_pths), len(r_pths))
            while len(l_pths) > minlen:
                l_pths.pop(-1)
            while len(m_pths) > minlen:
                m_pths.pop(-1)
            while len(r_pths) > minlen:
                r_pths.pop(-1)

            for L, M, R in zip(l_pths, m_pths, r_pths):
                plot_l = cv2.imread(L, 1)
                plot_m = cv2.imread(M, 1)
                plot_r = cv2.imread(R, 1)

                fram_l = cv2.imread(L.replace('plots', 'datasets'), 1)
                fram_m = cv2.imread(M.replace('plots', 'datasets'), 1)
                fram_r = cv2.imread(R.replace('plots', 'datasets'), 1)

                bc = cv2.BORDER_CONSTANT
                white = [255, 255, 255]
                fram_l = cv2.copyMakeBorder(
                    fram_l, 82, 83, 162, 163, bc, value=white)
                fram_m = cv2.copyMakeBorder(
                    fram_m, 82, 83, 162, 163, bc, value=white)
                fram_r = cv2.copyMakeBorder(
                    fram_r, 82, 83, 162, 163, bc, value=white)

                out = cv2.copyMakeBorder(
                    fram_l, 0, 480, 0, 1280, bc, value=white)

                for c in range(3):
                    out[0:480, 640:1280, c] = fram_m[:, :, c]
                    out[0:480, 1280:1920, c] = fram_r[:, :, c]
                    out[480:960, 0:640, c] = plot_l[:, :, c]
                    out[480:960, 640:1280, c] = plot_m[:, :, c]
                    out[480:960, 1280:1920, c] = plot_r[:, :, c]
                cv2.imwrite(L.replace(os.path.join(
                    trial_pth, '0'), trial_pth), out)
            shutil.rmtree(os.path.join(trial_pth, '0'))
            shutil.rmtree(os.path.join(trial_pth, '1'))
            shutil.rmtree(os.path.join(trial_pth, '2'))
