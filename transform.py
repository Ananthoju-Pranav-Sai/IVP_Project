import numpy as np
import torch


def fuse3D(opt, XYZ, masklogit, fuseTrans):
    XYZ = torch.transpose(XYZ, 1, 3)  # [B, 3V, W, H]
    XYZ = torch.transpose(XYZ, 2, 3)  # [B, 3V, H, W]
    masklogit = torch.transpose(masklogit, 1, 3)  # [B, V, W, H]
    masklogit = torch.transpose(masklogit, 2, 3)  # [B, V, H, W]
    # 2D to 3D coordinate transformation
    invKhom = torch.from_numpy(np.linalg.inv(opt.Khom2Dto3D))
    invKhomTile = torch.from_numpy(np.tile(invKhom, [opt.batchSize, opt.outViewN, 1, 1]).astype(np.float32))
    # viewpoint rigid transformation
    q_view = fuseTrans
    t_view = torch.from_numpy(np.tile([0, 0, -opt.renderDepth], [opt.outViewN, 1]).astype(np.float32))
    RtHom_view = transParamsToHomMatrix(q_view, t_view)
    RtHomTile_view = torch.from_numpy(np.tile(RtHom_view, [opt.batchSize, 1, 1, 1]).astype(np.float32))
    invRtHomTile_view = torch.from_numpy(np.linalg.inv(RtHomTile_view))
    # effective transformation
    RtHomTile = torch.matmul(invRtHomTile_view, invKhomTile)  # [B, V, 4, 4]
    RtTile = RtHomTile[:, :, :3, :]  # [B, V, 3, 4]
    # transform depth stack
    ML = torch.reshape(masklogit, [opt.batchSize, 1, -1])  # [B, 1, VHW]
    XYZhom = get3DhomCoord(XYZ, opt)  # [B, H, W, V, 4]
    XYZid = torch.matmul(RtTile, XYZhom.unsqueeze(-1))  # [B, V, 3, H, W, 1]
    # fuse point clouds
    XYZid = XYZid.permute(0, 2, 1, 3).reshape(opt.batchSize, 3, -1)  # [B, 3, VHW]
    return XYZid, ML  # [B, 1, VHW]


def render2D(opt, XYZod, ML, renderTrans):
    offsetDepth, offsetMaskLogit = 10.0, 1.0
    q_target = torch.from_numpy(np.reshape(renderTrans, [opt.novelN * opt.batchSize, 4]).astype(np.float32))
    t_target = torch.from_numpy(np.tile([0, 0, -opt.renderDepth], [opt.batchSize * opt.novelN, 1])).astype(np.float32)
    RtHom_target = torch.from_numpy(
        np.reshape(transParamsToHomMatrix(q_target, t_target), [opt.batchSize, opt.novelN, 4, 4]))

    # 3D to 2D coordinate transformation
    KupHom = opt.Khom2Dto3D * np.array([[opt.upscale], [opt.upscale], [1], [1]], dtype=np.float32)
    kupHomTile = torch.from_numpy(np.tile(KupHom, [opt.batchSize, opt.novelN, 1, 1]).astype(np.float32))

    # effective transformation
    RtHomTile = torch.matmul(kupHomTile, RtHom_target)  # [B, N, 4, 4]
    RtTile = RtHomTile[:, :, :3, :]  # [B, N, 3, 4]

    # transform depth stack
    XYZidHom = get3DhomCoord(XYZod, opt)  # [B, H, W, V, 4]
    XYZidHomTile = torch.from_numpy(np.tile(XYZidHom, [1, opt.novelN, 1, 1]).astype(np.float32))  # [B, N, 4, VWH]
    XYZnew = torch.matmul(RtTile, XYZidHomTile)  # [B, N, 3, VWH]
    Xnew, Ynew, Znew = XYZnew[:, :, 0, :], XYZnew[:, :, 1, :], XYZnew[:, :, 2, :]  # [B, N, VWH]

    # Concatenate all viewpoints
    MLcat = torch.from_numpy(np.reshape(np.tile(ML, [1, opt.novelN, 1]), [-1]))
    XnewCat = torch.from_numpy(np.reshape(Xnew, [-1]))
    YnewCat = torch.from_numpy(np.reshape(Ynew, [-1]))
    ZnewCat = torch.from_numpy(np.reshape(Znew, [-1]))
    batchIdxCat, novelIdxCat, _ = np.meshgrid(range(opt.batchSize), range(opt.novelN),
                                              range(opt.outViewN * opt.outH * opt.outW), indexing='ij')
    batchIdxCat = torch.from_numpy(np.reshape(batchIdxCat, [-1]))
    novelIdxCat = torch.from_numpy(np.reshape(novelIdxCat, [-1]))

    # apply in-range masks
    XnewCatInt = np.to_int32(XnewCat)
    YnewCatInt = np.to_int32(YnewCat)
    maskInside = (XnewCatInt >= 0) & (XnewCatInt < opt.W) & (YnewCatInt >= 0) & (YnewCatInt < opt.upscale * opt.H)
    valueInt = np.stack([XnewCatInt, YnewCatInt, batchIdxCat, novelIdxCat], axis=1)
    valueFloat = np.stack([1 / (ZnewCat + offsetDepth + 1e-8), MLcat], axis=1)
    insideInt = valueInt[maskInside]
    insideFloat = valueFloat[maskInside]
    MLnewValid = np.transpose(insideFloat)[1]  # [VWH, N]

    # apply visible masks
    maskVisible = MLnewValid > 0
    visInt = insideInt[maskVisible]
    visFloat = insideFloat[maskVisible]
    invisInt = insideInt[~maskVisible]
    invisFloat = insideFloat[~maskVisible]
    XnewVis, YnewVis, batchIdxVis, novelIdxVis = np.moveaxis(visInt, 1, 0)  # [VWH, N]
    iZnewVis, MLnewVis = np.hsplit(visFloat, 2)
    XnewInvis, YnewInvis, batchIdxInvis, novelIdxInvis = np.moveaxis(invisInt, 1, 0)  # [VWH, N]
    MLnewInvis = np.transpose(invisFloat)[1]

    # map to unsampled inverse depth and mask (visible)
    scatterIdx = torch.from_numpy(np.stack([batchIdxVis, novelIdxVis, YnewVis, XnewVis], axis=1))
    scatterShape = torch.from_numpy(
        np.array([opt.batchSize, opt.novelN, opt.H * opt.upscale, opt.W * opt.upscale, 3], dtype=np.int32))
    countOnes = torch.from_numpy(np.ones_like(iZnewVis))
    scatteriZMLCnt = torch.from_numpy(np.stack([iZnewVis, MLnewVis, countOnes], axis=1))
    scatterIdx = scatterIdx.cpu().numpy()
    scatterShape = scatterShape.cpu().numpy()
    upNewiZMLCnt = torch.zeros(tuple(scatterShape), dtype=torch.float32)
    upNewiZMLCnt[scatterIdx[:, 0], scatterIdx[:, 1], scatterIdx[:, 2], scatterIdx[:, 3],
    :] = scatteriZMLCnt.cpu().numpy()

    upNewiZMLCnt = torch.from_numpy(
        np.reshape(upNewiZMLCnt, [opt.batchSize * opt.novelN, opt.H * opt.upscale, opt.W * opt.upscale, 3]))

    # downsample back to original size
    upNewiZMLCnt_tensor = torch.tensor(upNewiZMLCnt)
    newiZMLCnt_tensor = torch.nn.functional.max_pool2d(upNewiZMLCnt_tensor.permute(0, 1, 4, 2, 5, 3),
                                                    kernel_size=(opt.upscale, opt.upscale),
                                                    stride=(opt.upscale, opt.upscale),
                                                    padding=0)

    newiZMLCnt = torch.from_numpy(np.reshape(newiZMLCnt_tensor, [opt.batchSize, opt.novelN, opt.H, opt.W, 3]))
    newInvDepth, newMaskLogitVis, Collision = np.split(newiZMLCnt, 3, axis=4)

    # map to unsampled inverse depth and mask (invisible)
    scatterIdx = torch.from_numpy(np.stack([batchIdxInvis, novelIdxInvis, YnewInvis, XnewInvis], axis=1))
    scatterShape = torch.from_numpy(
        np.array([opt.batchSize, opt.novelN, opt.H * opt.upscale, opt.W * opt.upscale, 1], dtype=np.int32))
    scatterML = torch.from_numpy(np.stack([MLnewInvis], axis=1))
    scatterIdx = scatterIdx.cpu().numpy()
    scatterShape = scatterShape.cpu().numpy()
    upNewML = torch.zeros(tuple(scatterShape), dtype=torch.float32)
    upNewML[scatterIdx[:, 0], scatterIdx[:, 1], scatterIdx[:, 2], scatterIdx[:, 3], :] = scatterML.cpu().numpy()

    upNewML = torch.from_numpy(
        np.reshape(upNewML, [opt.batchSize * opt.novelN, opt.H * opt.upscale, opt.W * opt.upscale, 1]))

    # downsample back to original size
    upNewML = torch.tensor(upNewML)
    newML = torch.nn.functional.max_pool2d(upNewML.permute(0, 1, 4, 2, 5, 3),
                                        kernel_size=(opt.upscale, opt.upscale),
                                        stride=(opt.upscale, opt.upscale),
                                        padding=0)

    newMaskLogitInvis = np.reshape(newML, [opt.batchSize, opt.novelN, opt.H, opt.W, 1])

    # Combining visible/invisible
    newMaskLogit = torch.from_numpy(np.where(newMaskLogitVis > 0, np.where(newMaskLogitInvis < 0, newMaskLogitInvis,
                                                                        np.ones_like(newInvDepth) * (
                                                                            -offsetMaskLogit))))
    newDepth = 1 / (newInvDepth + 1e-8) - offsetDepth

    return newDepth, newMaskLogit, Collision


def quaternionToRotMatrix(q):
    qa, qb, qc, qd = np.hsplit(q, 4)
    Temp = torch.from_numpy(np.stack([[1 - 2 * (qc ** 2 + qd ** 2), 2 * (qb * qc - qa * qd), 2 * (qa * qc + qb * qd)],
                                   [2 * (qb * qc + qa * qd), 1 - 2 * (qb ** 2 + qd ** 2), 2 * (qc * qd - qa * qb)],
                                   [2 * (qb * qd - qa * qc), 2 * (qa * qb + qc * qd), 1 - 2 * (qb ** 2 + qc ** 2)]]))

    R = torch.transpose(Temp, 0, 2)
    R = torch.transpose(R, 1, 2)
    return R


def transParamsToHomMatrix(q, t):
    N = np.shape(q)[0]
    R = quaternionToRotMatrix(q)
    Rt = torch.from_numpy(np.concatenate([R, np.expand_dims(t, axis=2)], axis=2))
    hom_aug = torch.from_numpy(np.concatenate([np.zeros([N, 1, 3], np.ones([N, 1, 1]))], axis=2))
    RtHom = torch.from_numpy(np.concatenate([Rt, hom_aug], axis=1))
    return RtHom


def get3DhomCoord(XYZ, opt):
    ones = np.ones([opt.batchSize, opt.outViewN, opt.outH, opt.outW])
    XYZhom = torch.transpose(
        torch.from_numpy(np.reshape(np.concatenate([XYZ, ones], axis=1), [opt.batchSize, 4, opt.outViewN, -1]), 1, 2))
    return XYZhom  # [B,V,4,HW]


def get3DhomCoord2(XYZ, opt):
    ones = np.ones([opt.batchSize, 1, opt.outViewN * opt.outH * opt.outW])
    XYZhom = np.concatenate([XYZ, ones], axis=1)
    return XYZhom  # [B,4,VHW]
