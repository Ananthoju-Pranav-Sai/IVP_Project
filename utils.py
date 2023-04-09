import torch
from transform import render2D


def loss(opt, point_cloud, gt_point_cloud, rendertrans):
    depth, mask, collision = render2D(opt, point_cloud[0], point_cloud[1], rendertrans)
    diff = depth - gt_point_cloud[0]
    diff[collision == 1] = 0
    l1_loss = torch.sum(torch.abs(diff))
    bce_loss = torch.nn.functional.binary_cross_entropy(depth, gt_point_cloud[1])
    return l1_loss + opt.Lambda * bce_loss
