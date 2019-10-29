import multiprocessing
import os
import sys
import torch

from src.alignment.align_matrix import AlignMatrix
from src.clustering.within_trial import WTCluster
from src.models.tcn_embedder import TCNEmbedder
from src.models.vae_embedder import VAEEmbedder
from src.plot.umap_plot import UMAPPlot
from src.plot.reward_plot import RewardPlot
from src.plot.scatter_plot import ScatterPlot
from src.plot.vae_plot import VAEPlot
from src.train.train_tcn import TCNTrainer
from src.train.train_vae import VAETrainer
from src.utils.config import Config
from src.utils.tools import Tools
pyout = Tools.pyout
poem = Tools.tqdm_pbar

# select what operations to perform
TRAIN_TCN = 1  # train TCN on some dataset
TCN_EMBED = 1  # store TCN embeddings of some dataset
V_RED_FIT = 1  # fit pca on trained embeddings
VIZ_EMBED = 1  # vizualize validation embeddings
TEMP_SEGM = 1  # vizualize temporal segmentation of videos
BT_VISUAL = 1  # make scatterplot .csv file based on dtw distance matrix
TRAIN_VAE = 1  # train vae for each perspective
VAE_EMBED = 1  # compute vae embeddings for val set
ALIGNMENT = 1  # allign all pairs of trials in val set
VAE_VIZUA = 1  # visualize vae reconstruction based on alignments
VZUAL_REW = 1  # visuzalize reward progression
HANDBRAKE = 1  # handbrake-cli to make outputs compatible with browser video
if VIZ_EMBED:  # can't do this without fitting to find scaling hyperparameters
    V_RED_FIT = True


# select datasets
roots = []
roots.append('/media/roblaundry/folding_full')
roots.append('/media/roblaundry/folding_single')
roots.append('/media/roblaundry/grasping')
roots.append('/media/roblaundry/grasping_reverse')
roots.append('/media/roblaundry/pouring')

os.system('clear')
Tools.pyout("STARTING PIPELINE")

devices = (torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
           torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


for root in roots:
    if Config.SERVER:
        save_root = Tools.fname(root)

    # train TCN on specified dataset
    if TRAIN_TCN:
        Tools.pyout("TRAIN_TCN")
        STATE_DICT_PATHS = (None, None)
        TRAIN_ROOT = os.path.join(root, 'train')
        VAL_ROOT = os.path.join(root, 'val')
        SAVE_LOC = os.path.join(save_root, 'tcn')
        EPOCHS = 100
        trainer = TCNTrainer(
            devices=devices,
            state_dict_paths=STATE_DICT_PATHS,
            train_root=TRAIN_ROOT,
            val_root=VAL_ROOT,
            save_loc=SAVE_LOC)
        trainer.train_loop(0, EPOCHS)
        del trainer

    if TCN_EMBED:
        Tools.pyout("TCN_EMBED")
        try:
            STATE_DICT_PATHS = (
                os.path.join('./res/models', save_root, 'tcn/gpu0.pth'),
                os.path.join('./res/models', save_root, 'tcn/gpu1.pth'))
            ROOTS = (
                os.path.join(root, 'train'),
                os.path.join(root, 'val'))
            embedder = TCNEmbedder(
                devices=devices,
                state_dict_paths=STATE_DICT_PATHS,
                roots=ROOTS)
            embedder.embed()
        except Exception as e:
            Tools.pyout(e, force=True)
            raise e
            sys.exit(1)

    plot = None
    if V_RED_FIT:
        Tools.pyout("V_PCA_FIT")
        plot = UMAPPlot(
            data_root=os.path.join(root, 'val'), n_components=2)
        plot.save_model(os.path.join('./res/models', save_root, 'reduce'))

    if VIZ_EMBED:
        Tools.pyout("VIZ_EMBED")
        OUTPUT_FOLDER = os.path.join(root, 'results/embeddings')
        if plot is None:
            plot = UMAPPlot(joblib_path=os.path.join(
                './res/models', save_root, 'reduce/reduction.joblib'))
        for trial_folder in poem(Tools.list_dirs(os.path.join(root, 'val')),
                                 "PLOTTING TCN EMBEDDINGS"):
            plot.visualize(trial_folder, OUTPUT_FOLDER)
        Tools.ffmpeg(OUTPUT_FOLDER)

    if TEMP_SEGM:
        Tools.pyout("TEMP_SEGM")
        OUTPUT_FOLDER = os.path.join(root, 'results/WT')
        if 'manual_folding' in root:
            n_clusters = 6
        elif 'pouring' in root:
            n_clusters = 5
        else:
            n_clusters = 3
        clusterer = WTCluster(
            joblib_path=os.path.join(
                './res/models', save_root, 'reduce/reduction.joblib'),
            n_clusters=n_clusters,
            averaged=True)
        for trial_folder in poem(Tools.list_dirs(os.path.join(root, 'val')),
                                 "CLUSTERING"):
            clusterer.visualize(trial_folder, OUTPUT_FOLDER)
        Tools.ffmpeg(OUTPUT_FOLDER)

    if BT_VISUAL:
        if 'pouring' in root:
            n_clusters = 2
        elif 'grasping' in root:
            n_clusters = 4
        else:
            n_clusters = 6
        _ = ScatterPlot(data_root=os.path.join(root, 'val'),
                        n_clusters=n_clusters)

    if TRAIN_VAE:
        Tools.pyout("TRAIN_VAE")
        STATE_DICT_PATH = None
        TRAIN_ROOT = os.path.join(root, 'train')
        VAL_ROOT = os.path.join(root, 'val')
        SAVE_LOC = os.path.join(save_root, 'vae')
        POS = tuple([Tools.fname(f)
                     for f in Tools.list_dirs(Tools.list_dirs(TRAIN_ROOT)[0])])
        EPOCHS = 1
        for pos in poem(POS, "TRAINING VAE"):
            trainer = VAETrainer(
                device=devices[0],
                state_dict_path=STATE_DICT_PATH,
                train_root=TRAIN_ROOT,
                val_root=VAL_ROOT,
                save_loc=os.path.join(save_root, 'vae', pos),
                pos=pos)
            trainer.loop(EPOCHS)
            del trainer

    if VAE_EMBED:
        Tools.pyout("VAE_EMBED")
        ROOT = os.path.join(root, 'val')
        POS = tuple([Tools.fname(f)
                     for f in Tools.list_dirs(Tools.list_dirs(ROOT)[0])])
        for pos in poem(POS, "STORING VAE EMBEDDINGS"):
            embedder = VAEEmbedder(
                device=devices[0],
                state_dict_path=os.path.join(
                    './res/models', save_root, 'vae', pos, 'vae_mdl.pth'),
                root=ROOT,
                pos=pos)
            embedder.embed()

    if ALIGNMENT:
        ROOT = os.path.join(root, 'val')
        AM = AlignMatrix(ROOT)
        AM.compute()

    if VAE_VIZUA:
        Tools.pyout("VAE_VIZUA")
        ROOT = os.path.join(root, 'val')
        STATE_DICT_ROOT = os.path.join('./res/models', save_root, 'vae')
        OUTPUT_FOLDER = os.path.join(root, 'results/VAE')
        plot = VAEPlot(device=devices[0],
                       state_dict_root=STATE_DICT_ROOT, root=ROOT)
        for trial_folder in poem(Tools.list_dirs(ROOT),
                                 "VIZUALIZING RECONSTRUCTION"):
            plot.visualize(trial_folder, OUTPUT_FOLDER)
        Tools.ffmpeg(OUTPUT_FOLDER)

    if VZUAL_REW:
        Tools.pyout("VIZUALIZE REWARD PROGRESSION")
        ROOT = os.path.join(root, 'val')
        OUTPUT_FOLDER = os.path.join(root, 'results/REWARD')
        plot = RewardPlot(ROOT)
        for trial_folder in poem(Tools.list_dirs(ROOT), "VIZUALIZING REWARD"):
            plot.visualize(trial_folder, OUTPUT_FOLDER)
        Tools.ffmpeg(OUTPUT_FOLDER)

    if HANDBRAKE:
        ROOT = os.path.join(root, 'results')
        tasks = []
        for path, dirs, files in os.walk(ROOT):
            for f in files:
                if f.endswith('.mp4'):
                    tasks.append(os.path.join(path, f))
        pool = multiprocessing.Pool(Config.N_WORKERS)
        for _ in poem(pool.imap_unordered(Tools.handbrake, tasks),
                      "HANDBRAKE", total=len(tasks)):
            pass


Tools.exit()
