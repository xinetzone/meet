import numpy as np


class Box:
    '''
    corner: (xmin,ymin,xmax,ymax)
    '''

    def __init__(self, corner):
        self._corner = corner

    @property
    def corner(self):
        return self._corner

    @corner.setter
    def corner(self, new_corner):
        self._corner = new_corner

    @property
    def w(self):
        '''
        计算 bbox 的 宽
        '''
        return self.corner[2] - self.corner[0] + 1

    @property
    def h(self):
        '''
        计算 bbox 的 高
        '''
        return self.corner[3] - self.corner[1] + 1

    @property
    def area(self):
        '''
        计算 bbox 的 面积
        '''
        return self.w * self.h

    @property
    def whctrs(self):
        '''
        计算 bbox 的 中心坐标
        '''
        xctr = self.corner[0] + (self.w - 1) * .5
        yctr = self.corner[1] + (self.h - 1) * .5
        return xctr, yctr

    def __and__(self, other):
        '''
        运算符：&，实现两个 box 的交集运算
        '''
        U = np.array([self.corner, other.corner])
        xmin, ymin, xmax, ymax = np.split(U, 4, axis=1)
        w = xmax.min() - xmin.max()
        h = ymax.min() - ymin.max()
        return w * h

    def __or__(self, other):
        '''
        运算符：|，实现两个 box 的并集运算
        '''
        I = self & other
        return self.area + other.area - I

    def IoU(self, other):
        '''
        计算 IoU
        '''
        I = self & other
        U = self | other
        return I / U