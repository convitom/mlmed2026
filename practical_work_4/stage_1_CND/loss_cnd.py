# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_iou_3d(box1, box2):
    """
    Compute 3D IoU between two boxes
    box1, box2: [x, y, z, diameter]
    """
    r1 = box1[3] / 2
    r2 = box2[3] / 2

    box1_min = box1[:3] - r1
    box1_max = box1[:3] + r1
    box2_min = box2[:3] - r2
    box2_max = box2[:3] + r2

    inter_min = np.maximum(box1_min, box2_min)
    inter_max = np.minimum(box1_max, box2_max)
    inter_size = np.maximum(0, inter_max - inter_min)
    inter_volume = np.prod(inter_size)

    box1_volume = (2 * r1) ** 3
    box2_volume = (2 * r2) ** 3
    union_volume = box1_volume + box2_volume - inter_volume

    return inter_volume / (union_volume + 1e-6)


class DetectionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.anchors = config['anchors']

        # Increase positive weight to boost recall
        self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.pos_weight = 5.0  # Can tune 5~10

        # Hard negative mining
        self.num_neg = config.get('num_neg', 800)

        # Regression weight (lower than cls for Stage 1)
        self.reg_weight = 0.5

    def forward(self, output, targets):
        batch_size = output.size(0)
        device = output.device

        total_cls_loss = 0
        total_reg_loss = 0
        total_pos = 0

        for b in range(batch_size):
            pred = output[b]  # [D,H,W,A,5]
            target_boxes = targets[b].cpu().numpy()

            D, H, W, A, _ = pred.shape

            pos_mask = torch.zeros(D, H, W, A, dtype=torch.bool, device=device)
            neg_mask = torch.zeros(D, H, W, A, dtype=torch.bool, device=device)
            target_reg = torch.zeros(D, H, W, A, 4, device=device)

            # Anchor Matching
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        grid_center = np.array([w*4+2, h*4+2, d*4+2])

                        for a_idx, anchor_size in enumerate(self.anchors):
                            anchor_box = np.concatenate([grid_center, [anchor_size]])

                            max_iou = 0
                            matched_gt = None

                            for gt in target_boxes:
                                iou = compute_iou_3d(anchor_box, gt)
                                if iou > max_iou:
                                    max_iou = iou
                                    matched_gt = gt

                            if max_iou > self.config['th_pos_train']:
                                pos_mask[d,h,w,a_idx] = True

                                dx = (matched_gt[0] - grid_center[0]) / anchor_size
                                dy = (matched_gt[1] - grid_center[1]) / anchor_size
                                dz = (matched_gt[2] - grid_center[2]) / anchor_size
                                dd = np.log(matched_gt[3] / anchor_size)

                                target_reg[d,h,w,a_idx] = torch.tensor(
                                    [dx, dy, dz, dd], device=device
                                )

                            elif max_iou < self.config['th_neg']:
                                neg_mask[d,h,w,a_idx] = True

            # Classification Loss
            pred_conf = pred[..., 0]

            pos_logits = pred_conf[pos_mask]
            neg_logits = pred_conf[neg_mask]

            # Positive loss
            if pos_logits.numel() > 0:
                pos_targets = torch.ones_like(pos_logits)
                pos_loss = self.cls_loss_fn(pos_logits, pos_targets)
                pos_loss = self.pos_weight * pos_loss.mean()
            else:
                pos_loss = torch.tensor(0., device=device)

            # Hard Negative Mining
            if neg_logits.numel() > 0:
                neg_targets = torch.zeros_like(neg_logits)
                neg_loss_all = self.cls_loss_fn(neg_logits, neg_targets)

                # Take top-k hardest negatives
                k = min(self.num_neg, neg_loss_all.numel())
                neg_loss_topk, _ = torch.topk(neg_loss_all, k)
                neg_loss = neg_loss_topk.mean()
            else:
                neg_loss = torch.tensor(0., device=device)

            cls_loss = pos_loss + neg_loss

            # Regression Loss
            if pos_mask.sum() > 0:
                pred_reg = pred[..., 1:]
                pred_reg_pos = pred_reg[pos_mask]
                target_reg_pos = target_reg[pos_mask]

                reg_loss = F.smooth_l1_loss(
                    pred_reg_pos,
                    target_reg_pos,
                    reduction='mean'
                )
            else:
                reg_loss = torch.tensor(0., device=device)

            total_cls_loss += cls_loss
            total_reg_loss += reg_loss
            total_pos += pos_mask.sum().item()

        total_cls_loss /= batch_size
        total_reg_loss /= batch_size

        total_loss = total_cls_loss + self.reg_weight * total_reg_loss

        return {
            "total_loss": total_loss,
            "cls_loss": total_cls_loss,
            "reg_loss": total_reg_loss,
            "num_pos": total_pos
        }


if __name__ == "__main__":
    # Test loss function
    from config_cnd import config
    
    criterion = DetectionLoss(config)
    
    # Dummy data
    batch_size = 2
    output = torch.randn(batch_size, 32, 32, 32, 3, 5)  # 3 anchors, 5 values
    
    # Dummy targets
    targets = [
        torch.tensor([[50, 60, 70, 8]]),  # 1 nodule
        torch.tensor([[30, 40, 50, 10], [80, 90, 100, 12]])  # 2 nodules
    ]
    
    losses = criterion(output, targets)
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Cls loss: {losses['cls_loss'].item():.4f}")
    print(f"Reg loss: {losses['reg_loss'].item():.4f}")
    print(f"Num positive: {losses['num_pos']}")
