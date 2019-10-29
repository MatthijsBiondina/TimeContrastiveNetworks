import cv2
from joblib import dump, load
import json
# import matplotlib
# from matplotlib import pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm
import umap
import warnings

from src.plot.cv_plot import CVPlot
from src.utils.tools import Tools
pyout = Tools.pyout
poem = Tools.tqdm_pbar


class UMAPPlot:
    def __init__(self,
                 joblib_path=None,
                 data_root=None,
                 n_components=2,
                 img_size=(720, 480)):
        self.img_size = img_size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if joblib_path is not None:
                self.reduction_model = self.load_model(joblib_path)
            else:
                self.reduction_model = self._init_dimred(
                    data_root, n_components)

    def visualize(self, in_folder, ou_folder):
        try:
            os.makedirs(ou_folder)
        except FileExistsError:
            pass
        # Tools.pyout(ou_folder)
        N_perspectives = len(Tools.list_dirs(in_folder))
        folder_name = in_folder.split('/')[-1]

        # define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(os.path.join(
            ou_folder, in_folder.split('/')[-1] + '.mp4'),
            fourcc, 16, (240 * N_perspectives, 480))
        Tools.pyout(os.path.join(
            ou_folder, in_folder.split('/')[-1] + '.mp4'))

        # init frame
        main_frm = np.zeros((480, 240 * N_perspectives, 3), dtype=np.uint8)

        # init plots
        plots = self._init_plots(N_perspectives)

        # load embedding dicts
        dicts = self._load_embedding_dicts(in_folder)

        # loop over all frames
        frames_exec = (
            'sorted(list(' +
            ' | '.join('set(dicts[{}])'.format(ii)
                       for ii in range(len(dicts))) +
            '))')
        frames = eval(frames_exec)
        max_frm = int(frames[-1].split('.')[0])
        min_frm = int(frames[0].split('.')[0])

        for frame in poem(frames, folder_name):
            for ii, pos in enumerate(sorted(Tools.list_dirs(in_folder))):
                try:
                    dic = dicts[ii]
                    side_str = pos.split('/')[-1]
                    lpx = 240 * (ii)
                    rpx = 240 * (ii + 1)
                    plot = plots[ii]

                    # add frame from perspective to main frame
                    frame_img = cv2.imread(
                        os.path.join(in_folder, side_str, frame))
                    frame_img = cv2.resize(frame_img, (240, 240))
                    main_frm[0:240, lpx:rpx, :] = np.copy(
                        frame_img[:, :, :3])

                    # get dimred from TCN embedding of frame
                    embedding = dic[frame]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        projection = tuple(
                            self.transform(self.reduction_model.transform(
                                np.array([embedding])).squeeze()))

                    # plot embedding in plot frame
                    plt_img = plot.plot(projection, prog=(int(
                        frame.split('.')[0]) - min_frm) / (max_frm - min_frm))

                    # add plot to main frame
                    main_frm[240:480, lpx:rpx, :] = np.copy(
                        plt_img[:, :, :3])
                except cv2.error:
                    pass

            # Tools.render(main_frm, ascii=True)

            writer.write(main_frm)
        writer.release()

    def _init_dimred(self, data_root, n_components):
        Y = []
        pbar = tqdm(Tools.search(data_root, ('embed_left.json',
                                             'embed_middle.json',
                                             'embed_right.json',
                                             'embed_steady.json',
                                             'embed_mobile.json')))
        pbar.set_description('load tcn embeddings')
        for embed_file in pbar:
            with open(embed_file, 'r') as f:
                embed_dict = json.load(f)
            for k, embedding in embed_dict.items():
                if np.isnan(embedding).any():
                    pyout(k, embed_file)
                    sys.exit(0)
                Y.append(embedding)
        Y = np.array(Y)
        return self._dimred_fit(Y, n_components=n_components)

    def _dimred_fit(self, data, n_components=2):
        Tools.pyout("FITTING DIMENSIONALITY REDUCTION MODEL")
        dimred = umap.UMAP()
        dimred.fit(data)

        data_ = dimred.transform(data)
        minmax = max(abs(np.min(data_)), abs(np.max(data_)))
        self.transform = lambda x: x / minmax
        Tools.pyout("--------------------------------> DONE")

        return dimred

    def _init_plots(self, N_plots):
        plots = []
        for _ in range(N_plots):
            plots.append(CVPlot(minx=-1.5, maxx=1.5, miny=-1.5, maxy=1.5))
        return plots

    def _load_embedding_dicts(self, folder):
        dicts = []
        for json_filename in sorted(Tools.list_files(
                folder, substr='embed_', end='.json')):
            with open(json_filename, 'r') as f:
                dicts.append(json.load(f))
        return dicts

    def load_model(self, dimred_model_file):
        return load(dimred_model_file)

    def save_model(self, save_loc):
        Tools.makedirs(save_loc)
        dump(self.reduction_model, os.path.join(save_loc, 'reduction.joblib'))
