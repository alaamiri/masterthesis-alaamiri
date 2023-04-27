import torch
from torch import nn


class ResNetBasicblock(nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True):
        super(ResNetBasicblock, self).__init__(locals())
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ReLUConvBN(C_in, C_out, 3, stride)
        self.conv_b = ReLUConvBN(C_out, C_out, 3)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        basicblock = self.conv_a(x, None)
        basicblock = self.conv_b(basicblock, None)
        residual = self.downsample(x) if self.downsample is not None else x
        return residual + basicblock

    def get_embedded_ops(self):
        return None

class ReLUConvBN(nn.Module):
    """
    Implementation of ReLU activation, followed by 2d convolution and then 2d batch normalization.
    """
    def __init__(self, C_in, C_out, kernel_size, stride=1, affine=True, bias=False, track_running_stats=True, **kwargs):
        super(ReLUConvBN, self).__init__()
        self.kernel_size = kernel_size
        pad = 0 if kernel_size == 1 else 1
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):
#         print('xxx --------->', x.size())
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += "{}x{}".format(self.kernel_size, self.kernel_size)
        return op_name

class Identity(nn.Module):
    """
    An implementation of the Identity operation.
    """

    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def get_embedded_ops(self):
        return None

class Zero(nn.Module):
    """
    Implementation of the zero operation. It removes
    the connection by multiplying its input with zero.
    """

    def __init__(self, stride, C_in=None, C_out=None, **kwargs):
        """
        When setting stride > 1 then it is assumed that the
        channels must be doubled.
        """
        super(Zero, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            else:
                return x[:, :, :: self.stride, :: self.stride].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1], shape[2], shape[3] = self.C_out, (shape[2] + 1) // self.stride, (shape[3] + 1) // self.stride
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

class AvgPool1x1(nn.Module):
    """
    Implementation of Avergae Pooling with an optional
    1x1 convolution afterwards. The convolution is required
    to increase the number of channels if stride > 1.
    """

    def __init__(
        self, kernel_size, stride, C_in=None, C_out=None, affine=True, **kwargs
    ):
        super(AvgPool1x1, self).__init__()
        self.stride = stride
        self.avgpool = nn.AvgPool2d(
            3, stride=stride, padding=1, count_include_pad=False
        )
        if stride > 1:
            assert C_in is not None and C_out is not None
            self.affine = affine
            self.C_in = C_in
            self.C_out = C_out
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.avgpool(x)
        if self.stride > 1:
            x = self.conv(x)
            x = self.bn(x)
        return x