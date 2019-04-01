from mxnet import gluon

from .bbox import MultiBox

class RPNAnchorGenerator(gluon.HybridBlock):
    r"""Anchor generator for Region Proposal Netoworks.

    Parameters
    ----------
    stride : int
        Feature map stride with respect to original image.
        This is usually the ratio between original image size and feature map size.
    base_size : int
        The width(and height) of reference anchor box.
    ratios : iterable of float
        The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    scales : iterable of float
        The areas of anchor boxes.
        We use the following form to compute the shapes of anchors:

        .. math::

            width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
            height_{anchor} = width_{anchor} \times ratio

    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.

    """

    def __init__(self, stride, base_size, ratios, scales, alloc_size, **kwargs):
        super().__init__(**kwargs)
        # 生成锚框初选模板，之后通过切片获取特征图的真正锚框
        anchors_generator = MultiBox(
            stride, base_size, ratios, scales, alloc_size)
        self.num_depth = anchors_generator.num_depth
        self.anchors = self.params.get_constant(
            'anchor_', anchors_generator.anchors)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, anchors):
        """Slice anchors given the input image shape.

        Inputs:
            - **x**: input tensor with (1 x C x H x W) shape.
        Outputs:
            - **out**: output anchor with (1, N, 4) shape. N is the number of anchors.
        """
        a = F.slice_like(anchors, x * 0, axes=(2, 3))
        return a.reshape((1, -1, 4))