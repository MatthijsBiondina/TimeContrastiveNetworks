import cv2
import numpy as np


class CVLine:
    def __init__(self,
                 shape=None,
                 minx=0,
                 maxx=1,
                 miny=0,
                 maxy=1):
        """
            shape in (y,x)
        """
        self.shape = shape + (3,)
        self.minx, self.maxx = minx, maxx
        self.miny, self.maxy = miny, maxy
        self.img = self._init_canvas()

        self.current_xy = None

    def plot(self, point):
        if self.current_xy is not None:
            cv2.line(self.img,
                     self._pt2xy(self.current_xy),
                     self._pt2xy(point),
                     (0, 0, 255), 2)
        self.current_xy = point
        return self.img

    def _pt2xy(self, pt):
        x = 20 + int((pt[0] - self.minx) / (self.maxx - self.minx) *
                     (self.shape[1] - 20))
        y = self.shape[0] - (20 + int(
            (pt[1] - self.miny) / (self.maxy - self.miny) *
            (self.shape[0] - 20)))
        return (x, y)

    def _init_canvas(self):
        img = np.full(self.shape, 255, dtype=np.uint8)
        # draw y-axis
        cv2.line(img,
                 self._pt2xy((self.minx, 0)),
                 self._pt2xy((self.maxx, 0)),
                 (150,) * 3)

        temp = np.full((20, 90, 3), 255, dtype=np.uint8)
        cv2.putText(temp, "reward ->", (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,) * 3, 1)
        temp = np.rot90(temp, k=1)
        dy = 60
        img[self.shape[0] // 2 - dy:self.shape[0] // 2 - dy + 90, 0:20, :] = \
            temp[:, :, :]
        cv2.putText(img,
                    str(self.miny),
                    (5, self.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150,) * 3)
        cv2.putText(img,
                    str(self.maxy),
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,) * 3)

        # Tools.render(temp, waitkey=5000)

        # draw x-axis
        cv2.line(img,
                 self._pt2xy((0, self.miny)),
                 self._pt2xy((0, self.maxy)),
                 (150,) * 3)
        cv2.putText(img,
                    "time ->",
                    (self.shape[1] // 2 - 30, self.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (150,) * 3,
                    1)

        return img
