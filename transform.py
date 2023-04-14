import numpy as np
import torch
import logging


def quaternionToRotMatrix(q):
    qa, qb, qc, qd = torch.unbind(q, dim=1)
    one = torch.stack([1 - 2 * (qc ** 2 + qd ** 2), 2 * (qb * qc - qa * qd), 2 * (qa * qc + qb * qd)])
    two = torch.stack([2 * (qb * qc + qa * qd), 1 - 2 * (qb ** 2 + qd ** 2), 2 * (qc * qd - qa * qb)])
    three = torch.stack([2 * (qb * qd - qa * qc), 2 * (qa * qb + qc * qd), 1 - 2 * (qb ** 2 + qc ** 2)])
    R = torch.permute(torch.stack([one, two, three]), dims=[2, 0, 1])
    return R


def transParamsToHomMatrix(q, t, device):
    N = q.shape[0]
    R = quaternionToRotMatrix(q)
    Rt = torch.cat([R, torch.unsqueeze(t, axis=-1)], axis=2)
    hom_aug = torch.ones([N, 1, 4], device=device)
    hom_aug[:, :, 0:3] = 0
    # hom_aug = torch.cat([np.zeros([N, 1, 3], np.ones([N, 1, 1]))], axis=2)
    RtHom = torch.cat([Rt, hom_aug], axis=1)
    return RtHom


def get3DhomCoord(XYZ, opt, device):
    ones = torch.ones([opt.batchSize, opt.outViewN, opt.outH, opt.outW], device=device)
    XYZhom = torch.transpose(torch.reshape(torch.cat([XYZ, ones], axis=1), [opt.batchSize, 4, opt.outViewN, -1]), 1, 2)
    return XYZhom  # [B,V,4,HW]


def get3DhomCoord2(XYZ, opt, device):
    ones = torch.ones([opt.batchSize, 1, opt.outViewN * opt.outH * opt.outW], device=device)
    XYZhom = torch.cat([XYZ, ones], axis=1)
    return XYZhom  # [B,4,VHW]


def fuse3D(opt, XYZ, masklogit, fuseTrans, device):
    # 2D to 3D coordinate transformation
    invKhom = torch.inverse(torch.tensor(opt.Khom2Dto3D, device=device))
    invKhomTile = torch.tile(invKhom, [opt.batchSize, opt.outViewN, 1, 1])
    # viewpoint rigid transformation
    q_view = torch.tensor(fuseTrans, device=device)
    t_view = torch.tile(torch.tensor([0, 0, -opt.renderDepth], device=device), [opt.outViewN, 1])
    RtHom_view = transParamsToHomMatrix(q_view, t_view, device)
    RtHomTile_view = torch.tile(RtHom_view, [opt.batchSize, 1, 1, 1])
    invRtHomTile_view = torch.inverse(RtHomTile_view)
    # effective transformation
    RtHomTile = torch.matmul(invRtHomTile_view, invKhomTile)  # [B, V, 4, 4]
    RtTile = RtHomTile[:, :, :3, :]  # [B, V, 3, 4]
    # transform depth stack
    ML = torch.reshape(masklogit, [opt.batchSize, 1, -1])  # [B, 1, VHW]
    XYZhom = get3DhomCoord(XYZ, opt, device)  # [B, H, W, V, 4]
    XYZid = torch.matmul(RtTile, XYZhom)  # [B, V, 3, H, W, 1]
    # fuse point clouds
    XYZid = XYZid.permute(0, 2, 1, 3).reshape(opt.batchSize, 3, -1)  # [B, 3, VHW]
    return XYZid, ML  # [B, 1, VHW]


def render2D(opt, XYZid, ML, renderTrans, device):
    offsetDepth, offsetMaskLogit = 10.0, 1.0
    q_target = torch.reshape(renderTrans, [opt.novelN * opt.batchSize, 4])
    t_target = torch.tile(torch.tensor([0, 0, -opt.renderDepth], device=device), [opt.batchSize * opt.novelN, 1])
    RtHom_target = torch.reshape(transParamsToHomMatrix(q_target, t_target, device), [opt.batchSize, opt.novelN, 4, 4])

    # 3D to 2D coordinate transformation
    KupHom = opt.Khom3Dto2D * np.array([[opt.upscale], [opt.upscale], [1], [1]], dtype=np.float32)
    kupHomTile = torch.tile(torch.tensor(KupHom, device=device), [opt.batchSize, opt.novelN, 1, 1])

    # effective transformation
    RtHomTile = torch.matmul(kupHomTile, RtHom_target)  # [B, N, 4, 4]
    RtTile = RtHomTile[:, :, :3, :]  # [B, N, 3, 4]

    # transform depth stack
    XYZidHom = get3DhomCoord2(XYZid, opt, device)  # [B, H, W, V, 4]
    XYZidHomTile = torch.tile(torch.unsqueeze(XYZidHom, 1), [1, opt.novelN, 1, 1])  # [B, N, 4, VWH]
    XYZnew = torch.matmul(RtTile, XYZidHomTile)  # [B, N, 3, VWH]
    Xnew, Ynew, Znew = XYZnew[:, :, 0, :], XYZnew[:, :, 1, :], XYZnew[:, :, 2, :]  # [B, N, VWH]

    # Concatenate all viewpoints
    MLcat = torch.reshape(torch.tile(ML, [1, opt.novelN, 1]), [-1])  # [B, N, VWH]
    XnewCat = torch.reshape(Xnew, [-1])
    YnewCat = torch.reshape(Ynew, [-1])
    ZnewCat = torch.reshape(Znew, [-1])
    batchIdxCat, novelIdxCat, _ = torch.meshgrid(torch.arange(start=0, end=opt.batchSize, step=1, device=device),
                                                 torch.arange(start=0, end=opt.novelN, step=1, device=device),
                                                 torch.arange(start=0, end=opt.outViewN * opt.outH * opt.outW, step=1, device=device),
                                                 indexing='ij')
    batchIdxCat = torch.reshape(batchIdxCat, [-1])
    novelIdxCat = torch.reshape(novelIdxCat, [-1])

    # apply in-range masks
    XnewCatInt = torch.round(XnewCat).to(torch.int32)
    YnewCatInt = torch.round(YnewCat).to(torch.int32)
    maskInside = (XnewCatInt >= 0) & (XnewCatInt < opt.upscale * opt.W) & (YnewCatInt >= 0) & (YnewCatInt < opt.upscale * opt.H)
    valueInt = torch.stack([XnewCatInt, YnewCatInt, batchIdxCat, novelIdxCat], dim=1)
    valueFloat = torch.stack([1 / (ZnewCat + offsetDepth + 1e-8), MLcat], dim=1)
    insideInt = valueInt[maskInside.cpu().numpy().astype(bool)]
    insideFloat = valueFloat[maskInside.cpu().numpy().astype(bool)]
    _, MLnewValid = torch.unbind(insideFloat, dim=1)  # [VWH, N]

    # apply visible masks
    maskVisible = (MLnewValid > 0).cpu().numpy().astype(bool)
    visInt = insideInt[maskVisible]
    visFloat = insideFloat[maskVisible]
    invisInt = insideInt[~maskVisible]
    invisFloat = insideFloat[~maskVisible]

    XnewVis, YnewVis, batchIdxVis, novelIdxVis = torch.unbind(visInt, dim=1)
    iZnewVis, MLnewVis = torch.unbind(visFloat, dim=1)
    XnewInvis, YnewInvis, batchIdxInvis, novelIdxInvis = torch.unbind(invisInt, dim=1)  # [VWH, N]
    _, MLnewInvis = torch.unbind(invisFloat, dim=1)

    # map to unsampled inverse depth and mask (visible)
    countOnes = torch.ones_like(iZnewVis)

    tmp = torch.ones(len(batchIdxVis), dtype=torch.int64, device=device)

    upNewiZMLCnt = torch.zeros(tuple([opt.batchSize, opt.novelN, opt.H * opt.upscale, opt.W * opt.upscale, 3]), device=device)
    upNewiZMLCnt.index_put_((batchIdxVis, novelIdxVis, YnewVis, XnewVis, 0 * tmp), iZnewVis, accumulate=True)
    upNewiZMLCnt.index_put_((batchIdxVis, novelIdxVis, YnewVis, XnewVis, tmp), MLnewVis, accumulate=True)
    upNewiZMLCnt.index_put_((batchIdxVis, novelIdxVis, YnewVis, XnewVis, 2 * tmp), countOnes, accumulate=True)

    upNewiZMLCnt = torch.reshape(upNewiZMLCnt,
                                 [opt.batchSize * opt.novelN, opt.H * opt.upscale, opt.W * opt.upscale, 3])
    upNewiZMLCnt = upNewiZMLCnt.permute([0, 3, 1, 2])

    # downsample back to original size
    newiZMLCnt_tensor = torch.nn.functional.max_pool2d(upNewiZMLCnt, kernel_size=opt.upscale, stride=opt.upscale,
                                                       padding=0)

    newiZMLCnt_tensor = newiZMLCnt_tensor.permute([0, 2, 3, 1])

    newiZMLCnt = torch.reshape(newiZMLCnt_tensor, [opt.batchSize, opt.novelN, opt.H, opt.W, 3])
    newInvDepth, newMaskLogitVis, Collision = torch.split(newiZMLCnt, 1, dim=4)

    upNewML = torch.zeros(tuple([opt.batchSize, opt.novelN, opt.H * opt.upscale, opt.W * opt.upscale, 1]), dtype=torch.float32, device=device)
    tmp = torch.ones(len(batchIdxInvis), dtype=torch.int64, device=device)
    upNewML.index_put_((batchIdxInvis, novelIdxInvis, YnewInvis, XnewInvis, 0 * tmp), MLnewInvis, accumulate=True)

    upNewML = torch.reshape(upNewML, [opt.batchSize * opt.novelN, opt.H * opt.upscale, opt.W * opt.upscale, 1])

    # downsample back to original size
    upNewML = upNewML.permute([0, 3, 1, 2])
    newML = torch.nn.functional.avg_pool2d(upNewML, kernel_size=opt.upscale, stride=opt.upscale, padding=0)
    newML = newML.permute([0, 2, 3, 1])

    newMaskLogitInvis = torch.reshape(newML, [opt.batchSize, opt.novelN, opt.H, opt.W, 1])

    # Combining visible/invisible
    newMaskLogit = torch.where(newMaskLogitVis > 0, newMaskLogitVis,
                               torch.where(newMaskLogitInvis < 0, newMaskLogitInvis,
                                           torch.ones_like(newInvDepth) * (-offsetMaskLogit)))

    newDepth = 1 / (newInvDepth + 1e-8) - offsetDepth

    newDepth = torch.reshape(newDepth, [opt.batchSize, opt.novelN, opt.outH, opt.outW])
    newMaskLogit = torch.reshape(newMaskLogit, [opt.batchSize, opt.novelN, opt.outH, opt.outW])
    Collision = torch.reshape(Collision, [opt.batchSize, opt.novelN, opt.outH, opt.outW])
    return newDepth, newMaskLogit, Collision
