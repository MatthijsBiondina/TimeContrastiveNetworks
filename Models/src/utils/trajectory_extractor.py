import json
import os

from src.utils.affine_transform import Affine_Fit
from src.utils.preprocessing import Prep
from src.utils.tools import Tools


class TrajectoryExtractor:
    def __init__(self):
        self.T_L = self._init_transformation(
            './res/datasets/grabbing/anchor_points/L.json')
        self.T_M = self._init_transformation(
            './res/datasets/grabbing/anchor_points/M.json')
        self.T_R = self._init_transformation(
            './res/datasets/grabbing/anchor_points/R.json')

    def extract(self, indir):
        trajectory = {}
        D_l, D_r, lvid_pth, rvid_pth, kinect_data, img_paths_l, img_paths_r = \
            self._load_dicts(indir)

        for img_pth_l, img_pth_r in zip(img_paths_l, img_paths_r):
            img_name = img_pth_l.split('/')[-1]  # right image has same name
            frame = str(int(img_name.split('.')[0]))

            # get folding person in both frames
            idx_L, person_L = self._get_person(D_l, img_name, 'L')
            idx_R, person_R = self._get_person(D_r, img_name, 'R')

            if person_L is not None or person_R is not None:
                trajectory[img_name] = self._get_wrist_coordinates(
                    lvid_pth, rvid_pth, frame, idx_L, idx_R, person_L,
                    person_R)

        return trajectory

    def _get_wrist_coordinates(
            self, lframe_data, rframe_data, idx_L, idx_R, person_L, person_R):
        """
        NB. the kinect recordings are mirrored. We swap left and right arm when
        we index lframe_data and rframe_data; ie. to get the left wrist, we
        use the index of what alphapose thought to be the right wrist and vice
        versa.
        """
        if person_L is None:  # use right frame to determine positions
            lwrist = self.T_R.transform(rframe_data[idx_R * 2])
            rwrist = self.T_R.transform(rframe_data[idx_R * 2 + 1])
        elif person_R is None:  # use left frame to determine positions
            lwrist = self.T_L.transform(lframe_data[idx_L * 2])
            rwrist = self.T_L.transform(lframe_data[idx_L * 2 + 1])
        else:  # if both frames have a valid person, check for both wrists
            # which frame has the highest score and use that one.
            if person_R['KP']['RWrist'][2] > person_L['KP']['RWrist'][2]:
                lwrist = self.T_R.transform(rframe_data[idx_R * 2])
            else:
                lwrist = self.T_L.transform(lframe_data[idx_L * 2])
            if person_R['KP']['LWrist'][2] > person_L['KP']['LWrist'][2]:
                rwrist = self.T_R.transform(rframe_data[idx_R * 2 + 1])
            else:
                rwrist = self.T_L.transform(lframe_data[idx_L * 2 + 1])

        return tuple(lwrist) + tuple(rwrist)

    def _get_person(self, D, img_name, side):
        try:
            peoples = [person for person in D[img_name] if
                       Prep.on_couch(person['KP']['LHip'], side) or
                       Prep.on_couch(person['KP']['RHip'], side)]
            if len(peoples) == 0:
                return None, None
            else:
                idx = max(range(len(peoples)),
                          key=lambda x: peoples[x]['score'])
                return idx, peoples[idx]
        except KeyError:
            return None, None

    def _load_dicts(self, indir):
        # load dictionaries of camera positions
        D_l = Prep.init_joint_dict(os.path.join(
            indir, '0', '3d', 'alphapose-results.json'))
        D_r = Prep.init_joint_dict(os.path.join(
            indir, '2', '3d', 'alphapose-results.json'))

        # load dictionaries of real positions relative to camera
        lvid_path = os.path.join(indir.replace(
            '/grabbing/', '/kinect-recordings-3/'), 'color-recording-left.avi')
        rvid_path = os.path.join(indir.replace(
            '/grabbing/', '/kinect-recordings-3/'),
            'color-recording-right.avi')
        xyz = None

        # load image paths of video and make equal length
        img_paths_l = sorted(Tools.list_files(os.path.join(indir, '0')))
        img_paths_r = sorted(Tools.list_files(os.path.join(indir, '2')))
        img_paths_l, img_paths_r = Prep.make_equal_length(
            (img_paths_l, img_paths_r))

        return D_l, D_r, lvid_path, rvid_path, xyz, img_paths_l, img_paths_r

    def _init_transformation(self, anchor_points_path):
        with open(anchor_points_path, 'r') as f:
            D = json.load(f)
        return Affine_Fit(D["from"], D["to"])
