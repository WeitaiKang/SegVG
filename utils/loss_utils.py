import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from utils.box_utils import xywh2xyxy, generalized_box_iou

def trans_vg_loss(batch_pred, batch_target, 
                  batch_pred_mask=None, batch_gt_region=None):
    """Compute the losses related to the bounding boxes, 
       including the L1 regression loss and the GIoU loss
    """
    L, bs, _ = batch_pred.shape
    batch_pred = batch_pred.view(L*bs, -1)
    batch_target = batch_target.unsqueeze(0).repeat(L, 1, 1).view(L*bs, -1)

    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none').view(L, bs, 4)
    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(batch_pred),
        xywh2xyxy(batch_target)
    )).view(L, bs) 

    losses = {}
    L1_factor = 5. 
    Giou_factor = 2. 
    later_factor = 1. 

    L, N, bs, _ = batch_pred_mask.shape
    batch_pred_mask = F.interpolate(batch_pred_mask.reshape(L*N, bs, 20, 20), size=(640, 640), mode='bilinear', align_corners=False).\
        reshape(L*N*bs, -1).sigmoid()

    batch_gt_region_ = torch.tensor(
        batch_gt_region.view(1, 1, bs, -1).expand(L, N, bs, 640*640).reshape(L*N*bs, -1)
        , dtype=torch.float32)

    focalbceloss, pos_factor = focal_loss(batch_pred_mask, batch_gt_region_, 0.25, 2)
    diceloss = dice_loss(batch_pred_mask, batch_gt_region_, ep=1e-8)
    losses['BCEloss'] = (focalbceloss).sum() / bs * later_factor #* 20
    losses['DICEloss'] = (diceloss * pos_factor).sum() / bs * later_factor
    pos_factor = pos_factor.view(L,N,bs).mean(dim=1)

    losses['loss_bbox'] = (loss_bbox * pos_factor.unsqueeze(-1)).sum() / bs * L1_factor
    losses['loss_giou'] = (loss_giou * pos_factor).sum() / bs * Giou_factor

    return losses

def dice_loss(predictive, batch_gt_region, ep=1e-8):
    intersection = 2 * torch.sum(predictive * batch_gt_region, dim=-1) + ep
    union = torch.sum(predictive, dim=-1) + torch.sum(batch_gt_region, dim=-1) + ep
    loss = 1 - intersection / union

    return loss


def focal_loss(prob, batch_gt_region, alpha: float = 0.25, gamma: float = 2):
    BCEloss = F.binary_cross_entropy(prob, batch_gt_region, reduction="none")
    p_t = prob * batch_gt_region + (1 - prob) * (1 - batch_gt_region)
    
    BCEloss = BCEloss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * batch_gt_region + (1 - alpha) * (1 - batch_gt_region)
        BCEloss = alpha_t * BCEloss

    pos_factor = (prob * batch_gt_region).sum(dim=-1) / batch_gt_region.sum(dim=-1).clamp(min=1)
    pos_factor = ((1 - pos_factor) ** gamma)
    
    return torch.mean(BCEloss, dim=-1), pos_factor + 1