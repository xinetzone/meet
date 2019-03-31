# Faster RCNN 学习与实现

- [论文](https://github.com/XinetAI/CVX/blob/master/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/papers/Faster_R-CNN_2017.pdf)
- [论文翻译](https://www.jianshu.com/p/7adc34483e4a)

Faster R-CNN 主要分为两个部分：

- RPN（Region Proposal Network）生成高质量的 region proposal；
- Fast R-CNN 利用 region proposal 做出检测。

在论文中作者将 RPN 比作神经网络的**注意力机制**（"attention" mechanisms），告诉网络看哪里。为了更好的理解，下面简要的叙述论文的关键内容。

## RPN

- Input：任意尺寸的图像
- Output：一组带有目标得分的目标矩形 proposals

为了生成 region proposals，在基网络的最后一个卷积层 `x` 上滑动一个小网络。该小网络由一个 $3\times 3$ 卷积 `conv1` 和一对兄弟卷积（并行的）$1\times 1$ 卷积 `reg` 和 `cls` 组成。其中，`conv1` 的参数 `padding=1`，`stride=1` 以保证其不会改变输出的特征图的尺寸。`reg` 作为 box-regression 用来编码 box 的坐标，`cls` 作为 box-classifaction 用来编码每个 proposal 是目标的概率。详细内容见我的博客：[我的目标检测笔记](https://www.cnblogs.com/q735613050/p/10573794.html)。论文中把不同 scale 和 aspect ratio 的 $k$ 个 reference boxes（参数化的 proposal） 称作 **anchors**（锚点）。锚点是滑块的中心。

为了更好的理解 anchors，下面以 Python 来展示其内涵。

### 锚点

首先利用[COCO 数据集的使用](https://www.cnblogs.com/q735613050/p/8969452.html)中介绍的 API 来获取一张 COCO 数据集的图片及其标注。

先载入一些必备的包：

```python
import cv2
from matplotlib import pyplot as plt
import numpy as np

# 载入 coco 相关 api
import sys
sys.path.append(r'D:\API\cocoapi\PythonAPI')
from pycocotools.dataset import Loader
%matplotlib inline
```

利用 `Loader` 载入 val2017 数据集，并选择包含 'cat', 'dog', 'person' 的图片：

```python
dataType = 'val2017'
root = 'E:/Data/coco'
catNms = ['cat', 'dog', 'person']
annType = 'annotations_trainval2017'
loader = Loader(dataType, catNms, root, annType)
```

输出结果：

```sh
Loading json in memory ...
used time: 0.762376 s
Loading json in memory ...
creating index...
index created!
used time: 0.401951 s
```

可以看出，`Loader` 载入数据的速度很快。为了更加详细的查看 `loader`，下面打印出现一些相关信息：

```python
print(f'总共包含图片 {len(loader)} 张')
for i, ann in enumerate(loader.images):
    w, h = ann['height'], ann['width']
    print(f'第 {i+1} 张图片的高和宽分别为: {w, h}')
```

显示：

```sh
总共包含图片 2 张
第 1 张图片的高和宽分别为: (612, 612)
第 2 张图片的高和宽分别为: (500, 333)
```

下面以第 1 张图片为例来探讨 anchors。先可视化：

```python
img, labels = loader[0]
plt.imshow(img);
```

输出：

![](../images/det.png)

为了让特征图的尺寸大一点，可以将其 resize 为 (800, 800, 3)：

```python
img = cv2.resize(img, (800, 800))
print(img.shape)
```

输出：

```sh
(800, 800, 3)
```

下面借助 MXNet 来完成接下来的代码编程，为了适配 MXNet 需要将图片由 (h, w, 3) 转换为 (3, w, h) 形式。

```python
img = img.transpose(2, 1, 0)
print(img.shape)
```

输出：

```sh
(3, 800, 800)
```

由于卷积神经网络的输入是四维数据，故而，还需要：

```python
img = np.expand_dims(img, 0)
print(img.shape)
```

输出

```sh
(1, 3, 800, 800)
```

为了和论文一致，我们也采用 VGG16 网络（载入 [gluoncv](https://gluon-cv.mxnet.io/)中的权重）：

```python
from gluoncv.model_zoo import vgg16
net = vgg16(pretrained=True)  #  载入权重
```

仅仅考虑直至最后一层卷积层(去除池化层)的网络，下面查看网络的各个卷积层的输出情况：

```python
from mxnet import nd
imgs = nd.array(img)  # 转换为 mxnet 的数据类型
x = imgs
for layer in net.features[:29]:
    x = layer(x)
    if "conv" in layer.name:
        print(layer.name, x.shape) # 输出该卷积层的 shape
```

结果为：

```sh
vgg0_conv0 (1, 64, 800, 800)
vgg0_conv1 (1, 64, 800, 800)
vgg0_conv2 (1, 128, 400, 400)
vgg0_conv3 (1, 128, 400, 400)
vgg0_conv4 (1, 256, 200, 200)
vgg0_conv5 (1, 256, 200, 200)
vgg0_conv6 (1, 256, 200, 200)
vgg0_conv7 (1, 512, 100, 100)
vgg0_conv8 (1, 512, 100, 100)
vgg0_conv9 (1, 512, 100, 100)
vgg0_conv10 (1, 512, 50, 50)
vgg0_conv11 (1, 512, 50, 50)
vgg0_conv12 (1, 512, 50, 50)
```

由此，可以看出尺寸为 (800, 800) 的原图变为了 (50, 50) 的特征图（比原来缩小了 16 倍）。

### 感受野

上面的 16 不仅仅是针对尺寸为 (800, 800)，它适用于任意尺寸的图片，因为 16 是特征图的一个像素点的感受野（receptive ﬁeld ）。

[^1]: Lenc K, Vedaldi A. R-CNN minus R.[J]. british machine vision conference, 2015.

感受野的大小是如何计算的？我们回忆卷积运算的过程，便可发现感受野的计算恰恰是卷积计算的逆过程（参考[感受野计算](papers/paper005.pdf)[^1]）。

记 $F_k, S_k, P_k$ 分别表示第 $k$ 层的卷积核的高(或者宽)、移动步长（stride）、Padding 个数；记 $i_k$ 表示第 $k$ 层的输出特征图的高（或者宽）。这样，很容易得出如下递推公式：

$$
i_{k+1} = \lfloor \frac{i_{k}-F_{k}+2P_{k}}{s_{k}}\rfloor + 1
$$

其中 $k \in \{1, 2, \cdots\}$，且 $i_0$ 表示原图的高或者宽。令 $t_k = \frac{F_k - 1}{2} - P_k$，上式可以转换为

$$
(i_{k-1} - 1) = (i_{k} - 1) S_k + 2t_k
$$

反推感受野, 令 $i_1 = F_1$, 且$t_k = \frac{F_k -1}{2} - P_k$, 且 $1\leq j \leq L$, 则有

$$
i_0 = (i_L - 1)\alpha_L + \beta_L
$$

其中 $\alpha_L = \prod_{p=1}^{L}S_p$，且有：

$$
\beta_L = 1 + 2\sum_{p=1}^L (\prod_{q=1}^{p-1}S_q) t_p
$$

由于 VGG16 的卷积核的配置均是 kernel_size=(3, 3), padding=(1, 1)，同时只有在经过池化层才使得 $S_j = 2$，故而 $\beta_j = 0$，且有 $\alpha_L = 2^4 = 16$。

### 锚点的计算

在编程实现的时候，将感受野的大小使用 `base_size` 来表示。下面我们讨论如何生成锚框？为了计算的方便，先定义一个 `Box`：

```python
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
```

类 Box 实现了 bbox 的交集、并集运算以及 IoU 的计算。下面举一个例子来说明：

```python
bbox = [0, 0, 15, 15]  # 边界框
bbox1 = [5, 5, 12, 12] # 边界框
A = Box(bbox)  # 一个 bbox 实例
B = Box(bbox1) # 一个 bbox 实例
```

下面便可以输出 A 与 B 的高宽、中心、面积、交集、并集、Iou：

```python
print('A 与 B 的交集', str(A & B))
print('A 与 B 的并集', str(A | B))
print('A 与 B 的 IoU', str(A.IoU(B)))
print('A 的中心、高、宽以及面积', str(A.whctrs), A.h, A.w, A.area)
```

输出结果：

```sh
A 与 B 的交集 49
A 与 B 的并集 271
A 与 B 的 IoU 0.18081180811808117
A 的中心、高、宽以及面积 (7.5, 7.5) 16 16 256
```

考虑到代码的可复用性，将 `Box` 封装进入 app/detection/bbox.py 中。下面重新考虑 loader。首先定义一个转换函数：

```python
def getX(img):
    # 将 img (h, w, 3) 转换为 (1, 3, w, h)
    img = img.transpose((2, 1, 0))
    return np.expand_dims(img, 0)
```

函数 `getX` 将图片由 (h, w, 3) 转换为 (1, 3, w, h)：

```python
img, label = loader[0]
img = cv2.resize(img, (800, 800)) # resize 为 800 x 800
X = getX(img)     # 转换为 (1, 3, w, h)
img.shape, X.shape
```

输出结果：

```sh
((800, 800, 3), (1, 3, 800, 800))
```

与此同时，获取特征图的数据：

```python
features = net.features[:29]
F = features(imgs)
F.shape
```

输出：

```sh
(1, 512, 50, 50)
```

接着需要考虑如何将特征图 F 映射回原图？

### 全卷积（FCN）：将特征图映射回原图

faster R-CNN 中的 FCN 仅仅是有着 FCN 的特性，并不是真正意义上的卷积。faster R-CNN 仅仅是借用了 FCN 的思想来实现将特征图映射回原图的目的，同时将输出许多锚框。

特征图上的 1 个像素点的感受野为 $16\times 16$，换言之，特征图上的锚点映射回原图的感受区域为 $16 \times 16$，论文称其为 reference box。下面相对于 reference box 依据不同的尺度与高宽比例来生成不同的锚框。

```python
base_size = 2**4  # 特征图的每个像素的感受野大小
scales = [8, 16, 32]  # 锚框相对于 reference box 的尺度
ratios = [0.5, 1, 2]  # reference box 与锚框的高宽的比率（aspect ratios）
```

其实 reference box 也对应于论文描述的 window（滑动窗口），这个之后再解释。我们先看看 scales 与 ratios 的具体含义。

为了更加一般化，假设 reference box 图片高宽分别为 $h, w$，而锚框的高宽分别为 $h_1, w_1$，形式化 scales 与 ratios 为公式 1：

$$
\begin{cases}
\frac{w_1 h_1}{wh} = s^2\\
\frac{h_1}{w_1} = \frac{h}{w} r \Rightarrow \frac{h_1}{h} = \frac{w_1}{w} r
\end{cases}
$$

可以将上式转换为公式 2：

$$
\begin{cases}
\frac{w_1}{w} = \frac{s}{\sqrt{r}}\\
\frac{h_1}{h} = \frac{w_1}{w} r = s \sqrt{r}
\end{cases}
$$

同样可以转换为公式3：

$$
\begin{cases}
w_s = \frac{w_1}{s} = \frac{w}{\sqrt{r}}\\
h_s = \frac{h_1}{s} = h \sqrt{r}
\end{cases}
$$

基于公式 2 与公式 3 均可以很容易计算出 $w_1,h_1$. 一般地，$w=h$，公式 3 亦可以转换为公式 4：

$$
\begin{cases}
w_s = \sqrt{\frac{wh}{r}}\\
h_s = w_s r
\end{cases}
$$

gluoncv 结合公式 4 来编程，本文依据 3 进行编程。无论原图的尺寸如何，特征图的左上角第一个锚点映射回原图后的 reference box 的 bbox = (xmain, ymin, xmax, ymax) 均为 (0, 0, bas_size-1, base_size-1)，为了方便称呼，我们称其为 base_reference box。基于 base_reference box 依据不同的 s 与 r 的组合生成不同尺度和高宽比的锚框，且称其为 base_anchors。编程实现：

```python
class MultiBox(Box):
    def __init__(self, base_size, ratios, scales):
        if not base_size:
            raise ValueError("Invalid base_size: {}.".format(base_size))
        if not isinstance(ratios, (tuple, list)):
            ratios = [ratios]
        if not isinstance(scales, (tuple, list)):
            scales = [scales]
        super().__init__([0]*2+[base_size-1]*2)  # 特征图的每个像素的感受野大小为 base_size
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

    def _generate_anchors(self, stride, alloc_size):
        # propagete to all locations by shifting offsets
        height, width = alloc_size  # 特征图的尺寸
        offset_x = np.arange(0, width * stride, stride)
        offset_y = np.arange(0, height * stride, stride)
        offset_x, offset_y = np.meshgrid(offset_x, offset_y)
        offsets = np.stack((offset_x.ravel(), offset_y.ravel(),
                            offset_x.ravel(), offset_y.ravel()), axis=1)
        # broadcast_add (1, N, 4) + (M, 1, 4)
        anchors = (self.base_anchors.reshape(
            (1, -1, 4)) + offsets.reshape((-1, 1, 4)))
        anchors = anchors.reshape((1, 1, height, width, -1)).astype(np.float32)
        return anchors
```

下面看看具体效果：

```python
base_size = 2**4  # 特征图的每个像素的感受野大小
scales = [8, 16, 32]  # 锚框相对于 reference box 的尺度
ratios = [0.5, 1, 2]  # reference box 与锚框的高宽的比率（aspect ratios）
A = MultiBox(base_size,ratios, scales)
A.base_anchors
```

输出结果：

```python
array([[ -84.,  -38.,   99.,   53.],
       [-176.,  -84.,  191.,   99.],
       [-360., -176.,  375.,  191.],
       [ -56.,  -56.,   71.,   71.],
       [-120., -120.,  135.,  135.],
       [-248., -248.,  263.,  263.],
       [ -36.,  -80.,   51.,   95.],
       [ -80., -168.,   95.,  183.],
       [-168., -344.,  183.,  359.]])
```

接着考虑将 base_anchors 在整个原图上进行滑动。比如，特征图的尺寸为 (5， 5) 而感受野的大小为 50，则 base_reference box 在原图滑动的情况（移动步长为 50）如下图：

```python
x, y = np.mgrid[0:300:50, 0:300:50]
plt.pcolor(x, y, x+y);  # x和y是网格,z是(x,y)坐标处的颜色值colorbar()
```

输出结果：

![](../images/grid.png)

原图被划分为了 25 个 block，每个 block 均代表一个 reference box。若 base_anchors 有 9 个，则只需要按照 stride = 50 进行滑动便可以获得这 25 个 block 的所有锚框（总计 5x5x9=225 个）。针对前面的特征图 F 有：

```python
stride = 16  # 滑动的步长
alloc_size = F.shape[2:]  # 特征图的尺寸
anchors = A._generate_anchors(stride, alloc_size).shape
```

输出结果：

```sh
(1, 1, 50, 50, 36)
```

即总共 $50\times 50 \times 9=22500$ 个锚点（anchors 数量庞大且必然有许多的高度重叠的框。）。至此，我们生成初始锚框的过程便结束了，同时很容易发现，anchors 的生成仅仅借助 Numpy 便完成了，这样做十分有利于代码迁移到 Pytorch、TensorFlow 等支持 Numpy 作为输入的框架。下面仅仅考虑 MXNet，其他框架以后再讨论。

回到本小节的 FCN 的讨论上来，作者