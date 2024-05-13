import torch.nn as nn
import torch.nn.functional as F
import math
import torch

# adapt from https://github.com/MIC-DKFZ/BraTS2017

def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False):
        super(ConvD, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes, norm)

    def forward(self, x):
        # if self.first:
            # x = self.maxpool(x)
        x = self.bn1(self.conv1(x))
        y = self.relu(self.bn2(self.conv2(x)))
        if self.dropout > 0:
            y = F.dropout3d(y, self.dropout)
        y = self.bn3(self.conv3(x))
        res = self.relu(x + y)
        if not self.first:
            y = self.maxpool(res)
        else: 
            y = res
        return y



class ConvBranch(nn.Module):
    def __init__(self, c=1, n=48, dropout=0.5, norm='gn'):
        super(ConvBranch, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2,
                mode='trilinear', align_corners=False)

        self.convd0 = ConvD(c, n, first=True)
        self.convd1 = ConvD(c,     n, dropout, norm)
        self.convd2 = ConvD(n,   2*n, dropout, norm)
        self.convd3 = ConvD(2*n, 4*n, dropout, norm)
        self.convd4 = ConvD(4*n, 8*n, dropout, norm)
        self.convd5 = ConvD(8*n,16*n, dropout, norm)

    def forward(self, x):
        x0 = self.convd0(x)

        x1 = self.convd1(x)
        # print("res1.shape: ", res1.shape)
        # print("x1.shape: ", x1.shape)

        x2 = self.convd2(x1)
        # print("res2.shape: ", res2.shape)
        # print("x2.shape: ", x2.shape)

        x3 = self.convd3(x2)
        # print("res3.shape: ", res3.shape)
        # print("x3.shape: ", x3.shape)

        x4 = self.convd4(x3)
        # print("res4.shape: ", res4.shape)
        # print("x4.shape: ", x4.shape)

        # x5 = self.convd5(x4)
        # print("res5.shape: ", res5.shape)
        # print("x5.shape: ", x5.shape)

        return [x0, x1, x2, x3, x4]