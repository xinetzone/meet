import numpy as np


class Box:
    '''
    corner: Numpy, List, Tuple, MXNet.nd, rotch.tensor
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
        assert isinstance(self.w, (int, float)), 'need int or float'
        xctr = self.corner[0] + (self.w - 1) * .5
        yctr = self.corner[1] + (self.h - 1) * .5
        return xctr, yctr

    def __and__(self, other):
        '''
        运算符：&，实现两个 box 的交集运算
        '''
        xmin = max(self.corner[0], other.corner[0])  # xmin 中的大者
        xmax = min(self.corner[2], other.corner[2])  # xmax 中的小者
        ymin = max(self.corner[1], other.corner[1])  # ymin 中的大者
        ymax = min(self.corner[3], other.corner[3])  # ymax 中的小者
        w = xmax - xmin
        h = ymax - ymin
        if w < 0 or h < 0: # 两个边界框没有交集
            return 0
        else:  
            return w * h

    def __or__(self, other):
        '''
        运算符：|，实现两个 box 的并集运算
        '''
        I = self & other
        if I == 0:
            return 0
        else:
            return self.area + other.area - I

    def IoU(self, other):
        '''
        计算 IoU
        '''
        I = self & other
        if I == 0:
            return 0
        else:
            U = self | other
            return I / U

class MultiBox(Box):
    def __init__(self, stride, base_size, ratios, scales, alloc_size):
        if not base_size:
            raise ValueError("Invalid base_size: {}.".format(base_size))
        if not isinstance(ratios, (tuple, list)):
            ratios = [ratios]
        if not isinstance(scales, (tuple, list)):
            scales = [scales]
        super().__init__([0]*2+[base_size-1]*2)  # 特征图的每个像素的感受野大小为 base_size
        # Number of anchors at each pixel
        self.num_depth = len(ratios) * len(scales)
        self._alloc_size = alloc_size
        self._stride = stride
        # reference box 与锚框的高宽的比率（aspect ratios）
        self._ratios = np.array(ratios)[:, None]
        self._scales = np.array(scales)     # 锚框相对于 reference box 的尺度

    @property
    def base_anchors(self):
        ws = np.round(self.w / np.sqrt(self._ratios))
        w = ws * self._scales
        h = w * self._ratios
        wh = np.stack([w.flatten(), h.flatten()], axis=1)
        wh = (wh - 1) * .5
        return np.concatenate([self.whctrs - wh, self.whctrs + wh], axis=1)

    @property
    def anchors(self):
        # propagete to all locations by shifting offsets
        height, width = self._alloc_size  # 特征图的尺寸
        offset_x = np.arange(0, width * self._stride, self._stride)
        offset_y = np.arange(0, height * self._stride, self._stride)
        offset_x, offset_y = np.meshgrid(offset_x, offset_y)
        offsets = np.stack((offset_x.ravel(), offset_y.ravel(),
                            offset_x.ravel(), offset_y.ravel()), axis=1)
        # broadcast_add (1, N, 4) + (M, 1, 4)
        anchors = (self.base_anchors.reshape(
            (1, -1, 4)) + offsets.reshape((-1, 1, 4)))
        anchors = anchors.reshape((1, 1, height, width, -1)).astype(np.float32)
        return anchors


class BoxTransform(Box):
    '''
    一组 bbox 的运算
    '''
    def __init__(self, F, corners):
        '''
        F 可以是 mxnet.nd, numpy, torch.tensor
        '''
        super().__init__(corners)
        self.corner = corners.T
        self.F = F

    def __and__(self, other):
        r'''
        运算符 `&` 交集运算
        '''
        xmin = self.F.maximum(self.corner[0].expand_dims(
            0), other.corner[0].expand_dims(1))  # xmin 中的大者
        xmax = self.F.minimum(self.corner[2].expand_dims(
            0), other.corner[2].expand_dims(1))  # xmax 中的小者
        ymin = self.F.maximum(self.corner[1].expand_dims(
            0), other.corner[1].expand_dims(1))  # ymin 中的大者
        ymax = self.F.minimum(self.corner[3].expand_dims(
            0), other.corner[3].expand_dims(1))  # ymax 中的小者
        w = xmax - xmin
        h = ymax - ymin
        cond = (w <= 0) + (h <= 0)
        I = self.F.where(cond, nd.zeros_like(cond), w * h)
        return I

    def __or__(self, other):
        r'''
        运算符 `|` 并集运算
        '''
        I = self & other
        U = self.area.expand_dims(0) + other.area.expand_dims(1) - I
        return self.F.where(U < 0, self.F.zeros_like(I), U)

    def IoU(self, other):
        '''
        交并比
        '''
        I = self & other
        U = self | other
        return self.F.where(U == 0, self.F.zeros_like(I), I / U)