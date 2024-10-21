# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist

from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils
import torch.nn.functional as F
from utils.box_utils import xywh2xyxy, generalized_box_iou
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, device: torch.device, 
                    epoch: int, max_norm: float = 0, writer= None, is_main_process=False, epochs=0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    n_iter_in_epoch = 0
    max_iter = len(data_loader)
    for batch in metric_logger.log_every(data_loader, print_freq, header, epochs, epoch):
        
        img_data, text_data, target, pos_region = batch
        current_iter = n_iter_in_epoch + epoch * max_iter
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        pos_region = pos_region.to(device)
        
        output, pred_mask = model(img_data, text_data)
        loss_dict = loss_utils.trans_vg_loss(output, target, pred_mask, pos_region)

        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v
                                      for k, v in loss_dict_reduced.items()}
        loss_value = sum(loss_dict_reduced_unscaled.values()).item()

        if is_main_process:
            writer.add_scalar('Loss/loss', loss_value, current_iter)
            for k, v in loss_dict_reduced_unscaled.items():
                writer.add_scalar('Loss/{}'.format(k), v, current_iter)
        n_iter_in_epoch += 1
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device, writer=None, epoch=None, is_main_process=False):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        img_data, text_data, target, _ = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        
        pred_boxes, _ = model(img_data, text_data)
        pred_boxes = pred_boxes[-1] # last layer bbox output

        miou, accu = eval_utils.trans_vg_eval_val(pred_boxes, target)
        
        metric_logger.update_v2('miou', torch.mean(miou), batch_size)
        metric_logger.update_v2('accu', accu, batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if is_main_process and writer is not None:
        writer.add_scalar('eval_acc', stats['accu'], epoch)
    return stats


@torch.no_grad()
def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    for idx, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target, _ = batch
        # batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        output, _ = model(img_data, text_data)
        output = output[-1] # last layer bbox output

        # Confidence Score
        # seg = F.interpolate(seg.view(batch_size, 1, 20, 20), size=(640, 640), mode = 'bilinear', align_corners=False).view(batch_size, -1)
        # conf = ((seg >= 0.35) * seg).sum(dim=-1) / (seg >= 0.35).sum(dim=-1)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0) # bs, 4
    gt_boxes = torch.cat(gt_box_list, dim=0)

    total_num, _ = gt_boxes.shape
    accu_num, iou = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)
    
    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1].unsqueeze(0))
    
    return accuracy