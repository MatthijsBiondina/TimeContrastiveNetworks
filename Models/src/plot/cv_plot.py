import cv2
from matplotlib import pyplot as plt
import numpy as np


class CVPlot:
    def __init__(self,
                 minx=-1.5, maxx=1.5, miny=-1.5, maxy=1.5,
                 cmap='gist_rainbow'):
        self.minx, self.maxx = minx, maxx
        self.miny, self.maxy = miny, maxy
        self.canvas_size = (240, 240, 3)
        self.img = np.full(self.canvas_size, 255, dtype=np.uint8)
        self.img = cv2.line(
            self.img,
            self._pt2xy((self.minx, 0)), self._pt2xy((self.maxx, 0)),
            (200,) * 3)
        self.img = cv2.line(
            self.img,
            self._pt2xy((0, self.miny)), self._pt2xy((0, self.maxy)),
            (200,) * 3)
        self.cmap = plt.cm.get_cmap(cmap)
        self.current_xy = None

    def plot(self, point, prog=0, color=None, scope=False):
        if color is None:
            c = self._prog2color(prog)
        else:
            c = color
        xy = self._pt2xy(point)
        cv2.circle(self.img, xy, 3, c, -1)

        if scope:
            img_ = np.copy(self.img)
            cv2.line(img_, (0, xy[1]), (self.canvas_size[0], xy[1]), c, 1)
            cv2.line(img_, (xy[0], 0), (xy[0], self.canvas_size[1]), c, 1)
            return img_
        else:
            return self.img

    def _prog2color(self, t):
        """
            convert progress [0,1] to color using cmap
        """
        return tuple(int(x * 255) for x in self.cmap(1 - t)[:3])

    def _pt2xy(self, pt):
        x = int((pt[0] - self.minx) / (self.maxx - self.minx) *
                self.canvas_size[0])
        y = self.canvas_size[1] - \
            int((pt[1] - self.miny) / (self.maxy - self.miny) *
                self.canvas_size[1])
        return (x, y)
