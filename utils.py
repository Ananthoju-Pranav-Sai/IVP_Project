import torch
from transform import render2D


def loss(opt, point_cloud, gt_point_cloud, rendertrans):
    depth, mask, _ = render2D(opt, point_cloud[0], point_cloud[1], rendertrans)
    ...
