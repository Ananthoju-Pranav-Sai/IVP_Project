import torch
from transform import render2D


def loss(opt, point_cloud, y, rendertrans):
    depth, mask, _ = render2D(opt, point_cloud, y, rendertrans)
