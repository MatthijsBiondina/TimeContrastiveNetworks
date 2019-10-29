import json
import multiprocessing
import os

from src.alignment.aligner import Aligner
from src.utils.config import Config
from src.utils.tools import Tools


class AlignMatrix:
    def __init__(self, root):
        self.root = root

    def compute(self):
        self._construct_matrix(self.root)

    def load(self):
        alignment = {}
        for trial_root in Tools.tqdm_pbar(
                Tools.list_dirs(self.root), "LOADING ALIGNMENT DATA"):
            with open(os.path.join(trial_root, 'alignment.json'), 'r') as f:
                alignment[trial_root] = json.load(f)

        return alignment

    def _load_matrix(self, root):
        for trial_root in Tools.tqdm_pbar(
                Tools.list_dirs(root), "LOADING ALIGNMENT DATA"):
            pass

    def _construct_matrix(self, root):

        def align(trial_root, anchor_root):
            A = Aligner()
            path = A.align(trial_root, anchor_root)
            return (trial_root, anchor_root, path)

        pool = multiprocessing.Pool(Config.N_WORKERS)
        lock = multiprocessing.Manager().Lock()
        tasks = []
        for trial_root in Tools.list_dirs(root):
            tasks.append((root, trial_root, lock))

        # try:
        for _ in Tools.tqdm_pbar(
                pool.imap_unordered(self._align, tasks),
                "ALIGNING (multiprocessing pool)", total=len(tasks)):
            pass

    def _align(self, args):
        A = Aligner()
        A.make_align_dict(*args)
        # return trial_root, anchor_root, path
