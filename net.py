# coding: utf-8
__author__ = 'Cclock'


import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):

    def __init__(self):

        super(FeatureExtractor, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3),
            nn.BatchNorm2d(256),
        )

    def forward(self, x):

        return self.feature(x)


class ClsBranch(nn.Module):

    def __init__(self, anchors=5, out=256):
        
        super(ClsBranch, self).__init__()
        self.anchors = anchors
        self.out = out
        self.cls_t = nn.Conv2d(256, 2 * anchors * out, kernel_size=3)
        self.cls_d = nn.Conv2d(256, out, kernel_size=3)

    def forward(self, z, x):

        z_f = self.cls_t(z)
        x_f = self.cls_d(x)
        ks = z_f.data.size()[-1]
        z_f = z_f.view(2 * self.anchors, self.out, ks, ks)
        out = F.conv2d(x_f, z_f)
        return out



class RegBranch(nn.Module):

    def __init__(self, anchors=5, out=256):

        super(RegBranch, self).__init__()
        self.anchors = anchors
        self.out = out
        self.reg_t = nn.Conv2d(256, 4 * anchors * out, kernel_size=3)
        self.reg_d = nn.Conv2d(256, out, kernel_size=3)

    def forward(self, z, x):

        z_f = self.reg_t(z)
        x_f = self.reg_d(x)
        ks = z_f.data.size()[-1]
        z_f = z_f.view(4 * self.anchors, self.out, ks, ks)
        out = F.conv2d(x_f, z_f)
        return out


class SiamRPN(nn.Module):

    def __init__(self, anchors=5, out=256):

        super(SiamRPN, self).__init__()
        self.anchors = anchors
        self.out = out
        self.feature_extractor = FeatureExtractor()
        self.cls_branch = ClsBranch(self.anchors, self.out)
        self.reg_branch = RegBranch(self.anchors, self.out)

    def forward(self, z, x):

        z_f = self.feature_extractor(z)
        x_f = self.feature_extractor(x)

        rcls = self.cls_branch(z_f, x_f)
        rreg = self.reg_branch(z_f, x_f)

        return rcls, rreg


if __name__ == '__main__':

    siam = SiamRPN()
    z = torch.randn((1, 3, 127, 127))
    x = torch.randn((1, 3, 255, 255))
    yc, yr = siam.forward(z, x)
    print(yc.shape, yr.shape)