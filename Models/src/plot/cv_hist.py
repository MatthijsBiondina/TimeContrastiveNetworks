# import colorsys
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time


class Column:
    def __init__(self,
                 width,
                 height,
                 label,
                 minval,
                 maxval,
                 n_bars=1):
        self.width = width
        self.height = height
        self.label = label
        self.maxval = maxval
        self.minval = minval

        self.btop = 32
        self.bbot = int((self.height - 10) / 4 * 3 + 5)
        self.barw = (self.width - 10) // n_bars

        self.canvas = self._init_canvas()

    def plot(self, values):
        img = np.copy(self.canvas)
        if isinstance(values, int):
            values = (values,)
        for ii, val in enumerate(values):
            try:
                h = int(
                    self.bbot - (self.bbot - self.btop) *
                    (val - self.minval) / (self.maxval - self.minval))
                cv2.rectangle(img,
                              (5 + ii * self.barw, h),
                              (5 + (ii + 1) * self.barw, self.bbot),
                              self._num2bgr(ii / (len(values) - 0.99)),
                              -1)
            except ZeroDivisionError:
                pass
            vstr = str(val)
            cv2.putText(img,
                        vstr if len(vstr) < 3 else vstr[:3],
                        (5 + ii * self.barw, self.btop - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0))

        return img

    def _init_canvas(self):
        img = np.full((self.height, self.width, 3), 255, np.uint8)

        # draw bars
        cv2.rectangle(img,
                      (4, self.btop - 1),
                      (self.width - 4, self.bbot + 1),
                      (100, 100, 100), 1)

        for ii, sublbl in enumerate(self.label.split('_')):
            cv2.putText(img,
                        sublbl,
                        (5, self.bbot + 5 + 12 + ii * 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (100, 100, 100))
        return img

    def _num2bgr(self, t):
        cmap = plt.cm.get_cmap('jet')
        rgb = cmap(t)[:3]
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))


class CVHist:
    def __init__(self,
                 shape=(240, 240),
                 labels=None,
                 minvls=None,
                 maxvls=None,
                 n_bars=1):
        self.canvas_size = shape + (3,)

        self.labels = labels
        self.minvls = minvls
        self.maxvls = maxvls
        self.n_bars = n_bars

        self.c_width = int(self.canvas_size[1] / (len(self.labels)))
        self.c_height = int(self.canvas_size[0])

        self.columns = []
        for ii, lbl in enumerate(self.labels):
            self.columns.append(Column(self.c_width, self.c_height, lbl,
                                       self.minvls[ii], self.maxvls[ii],
                                       n_bars=n_bars))

    def plot(self, values=None):
        if values is None:
            values = tuple(
                (max(0, self.minvls[ii]),) * self.n_bars
                for ii in range(len(self.labels)))

        img = np.full(self.canvas_size, 255, dtype=np.uint8)
        for ii, column in enumerate(self.columns):
            img[:, self.c_width * ii:self.c_width *
                (ii + 1), :] = column.plot(values[ii])
        return img
