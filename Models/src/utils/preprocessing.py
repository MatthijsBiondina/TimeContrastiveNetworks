import os
import subprocess
import sys
from src.utils.tools import Tools
from src.utils.config import Config
import cv2
import numpy as np
# import scipy as sp
from scipy import ndimage
import json
from matplotlib import cm
import warnings
import shutil
from src.pose.alphapose import AlphaPose
from src.utils.camera_transform import CameraTransform
from sklearn.linear_model import LinearRegression
import math as m


class Prep:
    @staticmethod
    def unpack_videos(indir, oudir, devices):
        found_positive = False
        for iter_path in sorted(Tools.list_dirs(indir)):
            if True or all(Prep.video_length(v) >= 60
                           for v in Tools.list_files(iter_path)) and \
                    len(Tools.list_files(iter_path)) > 0:
                found_positive = True
                print('Unpacking', iter_path)
                iter_targ_path = os.path.join(
                    oudir, iter_path.split('/')[-1])
                Tools.makedirs(iter_targ_path, delete=True)
                for vid_path in Tools.list_files(iter_path):
                    if 'color' in vid_path:
                        if 'left' in vid_path:
                            Prep.unpack_video(
                                vid_path, os.path.join(iter_targ_path, '0'))
                        elif 'middle' in vid_path:
                            Prep.unpack_video(
                                vid_path, os.path.join(iter_targ_path, '1'))
                        elif 'right' in vid_path:
                            Prep.unpack_video(
                                vid_path, os.path.join(iter_targ_path, '2'))
                        else:
                            print('Error unpacking', vid_path)
                    elif 'depth' in vid_path:
                        if 'left' in vid_path:
                            Prep.unpack_video(
                                vid_path,
                                os.path.join(iter_targ_path, '0/3d'),
                                rm_outline=True)
                        elif 'middle' in vid_path:
                            Prep.unpack_video(
                                vid_path,
                                os.path.join(iter_targ_path, '1/3d'),
                                rm_outline=True)
                        elif 'right' in vid_path:
                            Prep.unpack_video(
                                vid_path,
                                os.path.join(iter_targ_path, '2/3d'),
                                rm_outline=True)
                        else:
                            print('Error unpacking', vid_path)
            else:
                print('Ignore', iter_path)
        return found_positive
        # return True

    @staticmethod
    def unpack_video(source_path, oudir, rm_outline=False):
        Tools.makedirs(oudir)
        # crop = Config.crop_vals(source_path.split('/')[-1])
        i = 0
        cap = cv2.VideoCapture(source_path)
        print(oudir, end='')
        while True:
            try:
                _, frame = cap.read()
                # frame = frame[crop['n']:crop['s'], crop['w']:crop['e']]

                if frame is None:
                    break

                if rm_outline:
                    frame = Prep.rm_outline(frame)

                # frame = cv2.resize(frame, (368, 368),
                #                    interpolation=cv2.INTER_AREA)

                cv2.imwrite(os.path.join(
                    oudir, str(i).zfill(8) + '.jpg'), frame)
                if not Config.SERVER:
                    cv2.imshow('img', frame)
                    k = cv2.waitKey(1)
                    if k == ord('q'):
                        sys.exit(0)

                i += 1
                if not i % 100:
                    print(str(i).zfill(10), end='\r')
            except TypeError:
                cap.release()
                print()
                break

    @staticmethod
    def rm_outline(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        max_t = 30

        th = 30
        # buff = np.zeros((4,) + img.shape)
        # mod = np.zeros(img.shape)

        h, w = img.shape
        t = 0
        while np.any(img < th) and t < 5:
            t += 1

            V = img.copy()
            V[img < th] = 0
            VV = ndimage.gaussian_filter(V.astype(float), sigma=5.0) + 1e-3

            W = 0 * img.copy() + 1
            W[img < th] = 0
            WW = ndimage.gaussian_filter(W.astype(float), sigma=5.0) + 1e-3

            buff = (VV / WW).astype(np.uint8)

            img[img < th] = buff[img < th]
        return img

    @staticmethod
    def video_length(fileloc):
        command = ['ffprobe',
                   '-v', 'fatal',
                   '-show_entries',
                   'stream=width,height,r_frame_rate,duration',
                   '-of', 'default=noprint_wrappers=1:nokey=1',
                   fileloc, '-sexagesimal']
        ffmpeg = subprocess.Popen(
            command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = ffmpeg.communicate()
        if(err):
            print(err)
        out = out.decode('utf-8').split('\n')
        try:
            return int(out[3].split(':')[1]) * 60 + \
                int(float(out[3].split(':')[2]))
        except IndexError:
            return -float("inf")

    @staticmethod
    def alphapose(indir=None):
        ap = AlphaPose()
        for trial_path in Tools.list_dirs(indir):
            for pos_path in Tools.list_dirs(trial_path):
                ap.process_dir(
                    pos_path,
                    os.path.join(pos_path, '3d'))

    @staticmethod
    def crop_start_end(indir):
        D_l = Prep.init_joint_dict(os.path.join(
            indir, '0', '3d', 'alphapose-results.json'))
        D_r = Prep.init_joint_dict(os.path.join(
            indir, '2', '3d', 'alphapose-results.json'))

        img_paths_l = sorted(Tools.list_files(os.path.join(indir, '0')))
        img_paths_m = sorted(Tools.list_files(os.path.join(indir, '1')))
        img_paths_r = sorted(Tools.list_files(os.path.join(indir, '2')))

        img_paths_l, img_paths_r = Prep.make_equal_length(
            (img_paths_l, img_paths_r))

        t_ = None  # last positive frame
        splits = []
        for t, (img_pth_l, img_pth_r) in enumerate(
                zip(img_paths_l, img_paths_r)):
            img_name = img_pth_l.split('/')[-1]
            img_l = cv2.imread(img_pth_l)

            img_r = cv2.imread(img_pth_r)

            sitting = False
            try:
                for person in D_l[img_name]:
                    if Prep.on_couch(person['KP']['LHip'], 0) or \
                            Prep.on_couch(person['KP']['RHip'], 0):
                        sitting = True
                        break
            except KeyError:
                pass
            if not sitting:
                try:
                    for person in D_r[img_name]:
                        if Prep.on_couch(person['KP']['LHip'], 2) or \
                                Prep.on_couch(person['KP']['RHip'], 2):
                            sitting = True
                            break
                except KeyError:
                    pass

            if sitting:
                if t_ is None or t > t_ + 5:
                    splits.append([])
                splits[-1].append(img_name)
                t_ = t

                cv2.circle(img_l, (15, 15), 10, (0, 255, 0), thickness=-1)
            else:
                cv2.circle(img_l, (15, 15), 10, (0, 0, 255), thickness=-1)
            cv2.imshow('foo', np.concatenate((img_l, img_r), axis=1))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit(0)

        keep = max(splits, key=lambda x: len(x))

        # if len(keep) < 0.5 * len(img_paths_l):
        #     shutil.rmtree(indir)
        #     return False
        # else:
        #     for img_pth_l, img_pth_m, img_pth_r in zip(
        #             img_paths_l, img_paths_m, img_paths_r):
        #         img_name = img_pth_l.split('/')[-1]
        #         if img_name in keep:
        #             pass
        #             img_l = cv2.imread(img_pth_l)
        #             img_r = cv2.imread(img_pth_r)
        #             cv2.imshow('foo', np.concatenate((img_l, img_r), axis=1))
        #             if cv2.waitKey(1) & 0xFF == ord('q'):
        #                 print("no")
        #         else:
        #             pass
        #             # os.remove(img_pth_l)
        #             # os.remove(img_pth_m)
        #             # os.remove(img_pth_r)
        #             # os.remove(img_pth_l.replace(img_name, '3d/' + img_name))
        #             # os.remove(img_pth_m.replace(img_name, '3d/' + img_name))
        #             # os.remove(img_pth_r.replace(img_name, '3d/' + img_name))
        #     return True

    @staticmethod
    def wrist_dict(indir):
        datadir = indir.replace('kinect-recordings-3', 'grabbing')
        ou_dict = {}
        for trial_folder in Tools.list_dirs(indir):
            trial_name = trial_folder.split('/')[-1]

            D_l = Prep.init_joint_dict(os.path.join(
                datadir, trial_name, '0', '3d', 'alphapose-results.json'))
            D_m = Prep.init_joint_dict(os.path.join(
                datadir, trial_name, '1', '3d', 'alphapose-results.json'))
            D_r = Prep.init_joint_dict(os.path.join(
                datadir, trial_name, '2', '3d', 'alphapose-results.json'))

            ou_dict[os.path.join(trial_folder, 'color-recording-left.avi')] = \
                Prep.req_frames(D_l)
            ou_dict[os.path.join(
                trial_folder, 'color-recording-middle.avi')] = \
                Prep.req_frames(D_m)
            ou_dict[os.path.join(
                trial_folder, 'color-recording-right.avi')] = \
                Prep.req_frames(D_r)

        with open('/media/roblaundry/kinect-recordings-3/wrist_positions.json',
                  'w+') as f:
            json.dump(ou_dict, f, indent=2)

    @staticmethod
    def req_frames(D):
        # make dict for requesting real xyz
        ou = {}
        for img_name in sorted(D):
            try:
                img_id = int(img_name.split('.')[0])
                ou[img_id] = {'info': [], 'xy_img': []}
                # print(D)
                for i, person in enumerate(D[img_name]):
                    ou[img_id]['info'].append((i, 'RWrist'))
                    ou[img_id]['xy_img'].append(
                        (int(person['KP']['RWrist'][0]),
                            int(person['KP']['RWrist'][1])))

                    ou[img_id]['info'].append((i, 'LWrist'))
                    ou[img_id]['xy_img'].append(
                        (int(person['KP']['LWrist'][0]),
                            int(person['KP']['LWrist'][1])))
            except KeyError:
                pass

        return ou

    @staticmethod
    def extract_trajectory(indir):
        trajectory = {}

        # load dictionaries of camera projections
        D_l = Prep.init_joint_dict(os.path.join(
            indir, '0', '3d', 'alphapose-results.json'))
        D_r = Prep.init_joint_dict(os.path.join(
            indir, '2', '3d', 'alphapose-results.json'))

        # load dictionaries of real camera positions
        lvid_path = os.path.join(indir.replace(
            '/grabbing/', '/kinect-recordings-3/'), 'color-recording-left.avi')
        rvid_path = os.path.join(indir.replace(
            '/grabbing/', '/kinect-recordings-3/'), 'color-recording-right.avi')
        xyz = None

        # load image paths of video
        img_paths_l = sorted(Tools.list_files(os.path.join(indir, '0')))
        img_paths_r = sorted(Tools.list_files(os.path.join(indir, '2')))

        # if videos are not equal lenght, cut from start and end to make them
        # equal.
        img_paths_l, img_paths_r = Prep.make_equal_length(
            (img_paths_l, img_paths_r))

        # loop over all frames
        for img_pth_l, img_pth_r in zip(img_paths_l, img_paths_r):
            # left and right frame have same name (but in different folder)
            img_name = img_pth_l.split('/')[-1]
            frame = str(int(img_name.split('.')[0]))

            # for testing: display images
            img_l = cv2.imread(img_pth_l)
            img_r = cv2.imread(img_pth_r)

            # get person on couch in left view (if any)
            try:
                peoples = [person for person in D_l[img_name] if
                           Prep.on_couch(person['KP']['LHip'], 0) or
                           Prep.on_couch(person['KP']['RHip'], 0)]
                if len(peoples) == 0:
                    lperson = None
                else:
                    lperson_idx = max(range(len(peoples)),
                                      key=lambda x: peoples[x]['score'])
                    lperson = peoples[lperson_idx]
            except KeyError:
                pass

            # get person on couch in right view (if any)
            try:
                peoples = [person for person in D_r[img_name] if
                           Prep.on_couch(person['KP']['LHip'], 2) or
                           Prep.on_couch(person['KP']['RHip'], 2)]
                if len(peoples) == 0:
                    rperson = None
                else:
                    rperson_idx = max(range(len(peoples)),
                                      key=lambda x: peoples[x]['score'])
                    rperson = peoples[rperson_idx]
            except KeyError:
                pass

            # do not add frame to trajectory if both are None
            if lperson is not None or rperson is not None:
                # left wrist
                if lperson is None:  # so rperson is not None
                    # get left wrist from right frame
                    lwrist = Prep.affine_transform(
                        xyz[rvid_path][frame]['xyz'][rperson_idx * 2 + 1], 'R')
                    cv2.circle(
                        img_r,
                        (int(rperson['KP']['LWrist'][0]),
                         int(rperson['KP']['LWrist'][1])),
                        10, (0, 255, 0), -1)

                    # get right wrist from right frame
                    rwrist = Prep.affine_transform(
                        xyz[rvid_path][frame]['xyz'][rperson_idx * 2], 'R')
                    cv2.circle(
                        img_r,
                        (int(rperson['KP']['RWrist'][0]),
                         int(rperson['KP']['RWrist'][1])),
                        10, (0, 255, 0), -1)

                elif rperson is None:  # so lperson is not None
                    # get left wrist from left frame
                    lwrist = Prep.affine_transform(
                        xyz[lvid_path][frame]['xyz'][lperson_idx * 2 + 1], 'L')
                    cv2.circle(
                        img_l,
                        (int(lperson['KP']['LWrist'][0]),
                         int(lperson['KP']['LWrist'][1])),
                        10, (0, 255, 0), -1)

                    # get right wrist from left frame
                    rwrist = Prep.affine_transform(
                        xyz[lvid_path][frame]['xyz'][lperson_idx * 2], 'L')
                    cv2.circle(
                        img_l,
                        (int(lperson['KP']['RWrist'][0]),
                         int(lperson['KP']['RWrist'][1])),
                        10, (0, 255, 0), -1)

                else:
                    if rperson['KP']['LWrist'][2] > lperson['KP']['LWrist'][2]:
                        # get left wrist from right frame
                        lwrist = Prep.affine_transform(
                            xyz[rvid_path][frame]['xyz'][rperson_idx * 2 + 1],
                            'R')
                        cv2.circle(
                            img_l,
                            (int(lperson['KP']['LWrist'][0]),
                             int(lperson['KP']['LWrist'][1])),
                            10, (0, 0, 255), -1)
                        cv2.circle(
                            img_r,
                            (int(rperson['KP']['LWrist'][0]),
                             int(rperson['KP']['LWrist'][1])),
                            10, (0, 255, 0), -1)
                    else:
                        # get left wrist from left frame
                        lwrist = Prep.affine_transform(
                            xyz[lvid_path][frame]['xyz'][lperson_idx * 2 + 1],
                            'L')
                        cv2.circle(
                            img_l,
                            (int(lperson['KP']['LWrist'][0]),
                             int(lperson['KP']['LWrist'][1])),
                            10, (0, 255, 0), -1)
                        cv2.circle(
                            img_r,
                            (int(rperson['KP']['LWrist'][0]),
                             int(rperson['KP']['LWrist'][1])),
                            10, (0, 0, 255), -1)

                    if rperson['KP']['RWrist'][2] > lperson['KP']['RWrist'][2]:
                        # get right wrist from right frame
                        rwrist = Prep.affine_transform(
                            xyz[rvid_path][frame]['xyz'][rperson_idx * 2], 'R')
                        cv2.circle(
                            img_l,
                            (int(lperson['KP']['RWrist'][0]),
                             int(lperson['KP']['RWrist'][1])),
                            10, (0, 0, 255), -1)
                        cv2.circle(
                            img_r,
                            (int(rperson['KP']['RWrist'][0]),
                             int(rperson['KP']['RWrist'][1])),
                            10, (0, 255, 0), -1)
                    else:
                        # get right wrist from left frame
                        rwrist = Prep.affine_transform(
                            xyz[lvid_path][frame]['xyz'][lperson_idx * 2], 'L')
                        cv2.circle(
                            img_l,
                            (int(lperson['KP']['RWrist'][0]),
                             int(lperson['KP']['RWrist'][1])),
                            10, (0, 255, 0), -1)
                        cv2.circle(
                            img_r,
                            (int(rperson['KP']['RWrist'][0]),
                             int(rperson['KP']['RWrist'][1])),
                            10, (0, 0, 255), -1)
                trajectory[img_name] = [lwrist[0], lwrist[1], lwrist[2],
                                        rwrist[0], rwrist[1], rwrist[2]]

            # resize for easier rendering
            img_l = cv2.resize(
                img_l, (img_l.shape[1] // 2, img_l.shape[0] // 2))
            img_r = cv2.resize(
                img_r, (img_r.shape[1] // 2, img_r.shape[0] // 2))

            # display video
            Tools.render('DEBUG', np.concatenate((img_l, img_r), axis=1))

    @staticmethod
    def affine_transform(xyz, pos):
        return xyz

    @staticmethod
    def detect_wrists(indir):
        ct = CameraTransform()

        D_l = Prep.init_joint_dict(os.path.join(
            indir, '0', '3d', 'alphapose-results.json'))
        D_r = Prep.init_joint_dict(os.path.join(
            indir, '2', '3d', 'alphapose-results.json'))

        while True:
            img_paths_l = sorted(Tools.list_files(os.path.join(indir, '0')))
            img_paths_r = sorted(Tools.list_files(os.path.join(indir, '2')))

            img_paths_l, img_paths_r = Prep.make_equal_length(
                (img_paths_l, img_paths_r))

            trajectory = {}

            for img_pth_l, img_pth_r in \
                    zip(img_paths_l, img_paths_r):

                img_name = img_pth_l.split('/')[-1]

                img_l = cv2.imread(img_pth_l)
                img_r = cv2.imread(img_pth_r)
                img_ld = cv2.imread(img_pth_l.replace(
                    img_name, '3d/' + img_name))
                img_rd = cv2.imread(img_pth_r.replace(
                    img_name, '3d/' + img_name))

                posture = np.zeros((2, 12))
                weights = np.zeros((2, 12))
                on_couch = False

                for img, img_d, D, pos2real, pos in [
                        (img_l, img_ld, D_l, ct.left2real, 0),
                        (img_r, img_rd, D_r, ct.right2real, 2)]:
                    try:
                        contenders = []
                        for person in D[img_name]:
                            if Prep.on_couch(person['KP']['LHip'], pos) or \
                               Prep.on_couch(person['KP']['RHip'], pos):
                                contenders.append(person)
                        if len(contenders) > 0:
                            on_couch = True
                            person = max(contenders, key=lambda x: x['score'])

                            xprw = int(person['KP']['RWrist'][0])
                            yprw = int(person['KP']['RWrist'][1])
                            crw = person['KP']['RWrist'][2]
                            zprw = ct.gray_val(img_d, xprw, yprw)

                            xpre = int(person['KP']['RElbow'][0])
                            ypre = int(person['KP']['RElbow'][1])
                            cre = person['KP']['RElbow'][2]
                            zpre = ct.gray_val(img_d, xpre, ypre)

                            xplw = int(person['KP']['LWrist'][0])
                            yplw = int(person['KP']['LWrist'][1])
                            clw = person['KP']['LWrist'][2]
                            zplw = ct.gray_val(img_d, xplw, yplw)

                            xple = int(person['KP']['LElbow'][0])
                            yple = int(person['KP']['LElbow'][1])
                            cle = person['KP']['LElbow'][2]
                            zple = ct.gray_val(img_d, xple, yple)

                            xrrw, yrrw, zrrw = pos2real(xprw, yprw, zprw)
                            xrre, yrre, zrre = pos2real(xpre, ypre, zpre)
                            xrlw, yrlw, zrlw = pos2real(xplw, yplw, zplw)
                            xrle, yrle, zrle = pos2real(xple, yple, zple)

                            arx = m.atan(float(xrrw - xrre) /
                                         float(zrrw - zrre + 1e-42))
                            ary = m.atan(float(yrrw - yrre) /
                                         float(zrrw - zrre + 1e-42))
                            arz = m.atan(float(xrrw - xrre) /
                                         float(yrrw - yrre + 1e-42)) / 2

                            alx = m.atan(float(xrlw - xrle) /
                                         float(zrlw - zrle + 1e-42))
                            aly = m.atan(float(yrlw - yrle) /
                                         float(zrlw - zrle + 1e-42))
                            alz = m.atan(float(xrlw - xrle) /
                                         float(yrlw - yrle + 1e-42)) / 2

                            posture[0 if pos == 0 else 1, :] = np.array([xrlw, yrlw, zrlw,
                                                                         alx, aly, alz,
                                                                         xrrw, yrrw, zrrw,
                                                                         arx, ary, arz])
                            weights[0 if pos == 0 else 1, :] = np.array([clw, clw, clw,
                                                                         clw + cle, clw + cle, clw + cle,
                                                                         crw, crw, crw,
                                                                         crw + cre, crw + cre, crw + cre])

                            for IMG in [img, img_d]:

                                for xpw, ypw, zpw, xrw, yrw, zrw, \
                                    xpe, ype, zpe, xre, yre, zre in (
                                        (xprw, yprw, zprw, xrrw, yrrw, zrrw,
                                         xpre, ypre, zpre, xrre, yrre, zrre),
                                        (xplw, yplw, zplw, xrlw, yrlw, zrlw,
                                         xple, yple, zple, xrle, yrle, zrle)):

                                    measure = zrw  # ct.real2left(xr, yr, zr)
                                    color = Prep.scr2rgb(-measure / 140. + 0.5)

                                    cv2.line(IMG, (xpw, ypw),
                                             (xpe, ype), color)
                                    cv2.circle(IMG, (xpw, ypw), 3,
                                               color, thickness=-1)
                                    cv2.circle(IMG, (xpe, ype), 3,
                                               color, thickness=-1)
                    except KeyError:
                        pass

                if on_couch:
                    posture = np.average(posture, axis=0, weights=weights)
                    trajectory[img_name] = posture.copy().tolist()

                disp = np.concatenate((img_l, img_r), axis=1)

                if not Config.SERVER:
                    cv2.imshow('img', disp)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        sys.exit(0)
            # with open(os.path.join(indir, 'trajectory.json'), 'w+') as f:
            #     json.dump(trajectory, f, indent=2)
            break

    @staticmethod
    def init_joint_dict(path):
        with open(path) as f:
            D = json.load(f)
            joints = {}
            kp = 'keypoints'
            for d in D:
                keypoints = {
                    'LElbow': (d[kp][21], d[kp][22], d[kp][23]),
                    'RElbow': (d[kp][24], d[kp][25], d[kp][26]),
                    'LWrist': (d[kp][27], d[kp][28], d[kp][29]),
                    'RWrist': (d[kp][30], d[kp][31], d[kp][32]),
                    'LHip': (d[kp][33], d[kp][34], d[kp][35]),
                    'RHip': (d[kp][36], d[kp][37], d[kp][38])}

                try:
                    joints[d['image_id']].append(
                        {'score': d['score'], 'KP': keypoints})
                except KeyError:
                    joints[d['image_id']] = [
                        {'score': d['score'], 'KP': keypoints}]
        return joints

    @staticmethod
    def on_couch(X, pos):
        x, y, _ = X
        #
        if (pos == 0 or pos == 'L') and (
                y < Tools.line_x(Tools.line((480, 176), (811, 68)), x) or
                y < Tools.line_x(Tools.line((811, 68), (959, 215)), x) or
                y > Tools.line_x(Tools.line((959, 215), (572, 372)), x) or
                y > Tools.line_x(Tools.line((480, 176), (572, 372)), x)):
            return False
        elif (pos == 2 or pos == 'R') and (
                y < Tools.line_x(Tools.line((472, 211), (789, 243)), x) or
                y > Tools.line_x(Tools.line((321, 397), (736, 465)), x) or
                y < Tools.line_x(Tools.line((472, 211), (321, 397)), x) or
                y > Tools.line_x(Tools.line((789, 243), (736, 465)), x)):
            return False
        return True

        # # ORIGINAL FOLDING DATA VERSION
        # if pos == 0 and (
        #         y < Tools.line_x(Tools.line((480, 176), (811, 68)),x) - 0.5 * x + 134 or
        #         y > -0.62 * x + 251 or
        #         y < x - 145 or
        #         y > 1.62 * x - 42):
        #     return False
        # elif pos == 1 and (
        #         y < 313 or
        #         x < 90 or
        #         x > 292):
        #     return False
        # elif pos == 2 and (
        #         y > 0.3 * x + 123 or
        #         y < 0.26 * x + 49 or
        #         y > -2.94 * x + 828 or
        #         y < -1.52 * x + 291):
        #     return False
        # return True

    @staticmethod
    def scr2rgb(score, scale=3., cmap='seismic'):
        score = min(1., max(0., (score)))
        c = eval('cm.' + cmap + '(score)')
        return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))

    @staticmethod
    def make_equal_length(lists):
        minlen = min(len(x) for x in lists)
        for i, lst in enumerate(lists):
            while len(lst) > minlen:
                lst.pop(-1)
        return lists
