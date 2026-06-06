import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.postprocess import soft_argmax_normalized


def wing_loss(pred, target, w=10.0, epsilon=2.0):
    """
    Wing loss for coordinate regression.
    Reference: Feng et al., CVPR 2018.
    """
    diff = torch.abs(pred - target)
    C    = w - w * math.log(1.0 + w / epsilon)
    loss = torch.where(
        diff < w,
        w * torch.log(1.0 + diff / epsilon),
        diff - C,
    )
    return loss.mean()


def wing_loss_elementwise(pred, target, w=10.0, epsilon=2.0):
    """
    Element-wise Wing loss — returns [B, 4] without reducing,
    so per-sample losses can be sorted for Worst-K mining.
    """
    diff = torch.abs(pred - target)
    C    = w - w * math.log(1.0 + w / epsilon)
    return torch.where(
        diff < w,
        w * torch.log(1.0 + diff / epsilon),
        diff - C,
    )


class HeatmapLoss(nn.Module):
    """
    Balanced heatmap + coordinate loss with per-landmark weighting.

    lm1_coord_weight   : Wing loss scale for LM1 relative to LM2 (default 3.0).
    lm1_heatmap_weight : BCE + Dice scale for LM1 channel relative to LM2 (default 2.0).
    sep_min_dist       : Separation margin in normalised [0,1] space (default 0.15 ≈ 38px).
    sep_weight         : Strength of separation penalty (default 5.0).

    NOTE: lm_weights parameter kept for backward compatibility but its
    normalisation (/ w.mean()) has been removed — raw weights are used directly.
    hard_k: keep only the K highest-error samples per batch for the gradient.
    """

    def __init__(
        self,
        coord_weight=10.0,
        sep_weight=5.0,
        sep_min_dist=0.15,
        wing_w=0.02,
        wing_eps=0.005,
        lm_weights=None,
        hard_k=None,
        lm1_coord_weight=3.0,
        lm1_heatmap_weight=2.0,
    ):
        super().__init__()
        self.coord_weight      = coord_weight
        self.sep_weight        = sep_weight
        self.sep_min_dist      = sep_min_dist
        self.wing_w            = wing_w
        self.wing_eps          = wing_eps
        self.hard_k            = hard_k
        self.lm1_coord_weight  = lm1_coord_weight
        self.lm1_heatmap_weight = lm1_heatmap_weight

        # lm_weights kept for API compat; no longer normalised
        if lm_weights is not None:
            w = torch.tensor(lm_weights, dtype=torch.float32)
            self.register_buffer("lm_weights", w)
        else:
            self.lm_weights = None

    def _soft_dice(self, pred_logits, target):
        """Soft Dice on a single-channel slice [B,1,H,W]."""
        pred   = torch.sigmoid(pred_logits)
        smooth = 1e-6
        inter  = (pred * target).sum(dim=(2, 3))
        union  = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        return (1 - (2 * inter + smooth) / (union + smooth)).mean()

    def forward(self, pred, target):
        # ── 1. Per-channel heatmap losses ────────────────────────────────────
        bce_lm1 = F.binary_cross_entropy_with_logits(pred[:, 0:1], target[:, 0:1])
        bce_lm2 = F.binary_cross_entropy_with_logits(pred[:, 1:2], target[:, 1:2])
        loss_bce = self.lm1_heatmap_weight * bce_lm1 + 1.0 * bce_lm2

        dice_lm1 = self._soft_dice(pred[:, 0:1], target[:, 0:1])
        dice_lm2 = self._soft_dice(pred[:, 1:2], target[:, 1:2])
        loss_dice = self.lm1_heatmap_weight * dice_lm1 + 1.0 * dice_lm2

        # ── 2. Per-landmark coordinate loss (Wing, normalised [0,1] space) ──
        pred_sig    = torch.sigmoid(pred)
        pred_coords = soft_argmax_normalized(pred_sig)
        gt_coords   = soft_argmax_normalized(target)

        raw = wing_loss_elementwise(pred_coords, gt_coords,
                                    w=self.wing_w, epsilon=self.wing_eps)  # [B, 4]

        loss_lm1 = raw[:, :2].mean()
        loss_lm2 = raw[:, 2:].mean()

        # Per-sample loss for Worst-K mining uses weighted sum before reduction
        per_sample = (
            self.lm1_coord_weight * raw[:, :2].mean(dim=-1)
            + 1.0               * raw[:, 2:].mean(dim=-1)
        )   # [B]
        if self.hard_k is not None and per_sample.size(0) > self.hard_k:
            per_sample = per_sample.topk(self.hard_k).values

        loss_coord = per_sample.mean()

        # ── 3. Separation loss ───────────────────────────────────────────────
        p1       = pred_coords[:, :2]
        p2       = pred_coords[:, 2:]
        sep      = torch.norm(p1 - p2, dim=1).mean()
        loss_sep = torch.clamp(self.sep_min_dist - sep, min=0.0)

        total = (
            loss_bce
            + loss_dice
            + self.coord_weight * loss_coord
            + self.sep_weight   * loss_sep
        )

        return total, {
            "bce":   loss_bce.item(),
            "dice":  loss_dice.item(),
            "coord": loss_coord.item(),
            "sep":   loss_sep.item(),
        }