import torch
from transform import render2D


def loss(opt, point_cloud, gt_point_cloud, rendertrans, device):
    depth, mask, collision = render2D(opt, point_cloud[0], point_cloud[1], rendertrans, device)
    diff = depth - gt_point_cloud[0]
    l1_loss = torch.sum(torch.abs(torch.masked_select(diff, collision == 1)))/(opt.novelN*opt.batchSize)
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(mask, gt_point_cloud[1].to(torch.float32), reduction='sum')/(opt.novelN*opt.batchSize)
    return l1_loss, bce_loss
