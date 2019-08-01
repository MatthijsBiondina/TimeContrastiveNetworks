from src.labeling import Labeler
from src.utils.tools import Tools

IN_ROOT = ''

labeler = Labeler(npos=3,
                  labels=('isolated_grasping', 'unfold', 'flatten',
                          'folding_progress', 'stack'))

for trial in Tools.list_dirs(IN_ROOT):
    labeler.label_video(trial)
