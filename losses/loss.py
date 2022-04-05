import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELossWithQuant(nn.Module):
    def __init__(self, image_channels=159, codebook_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.register_buffer(
            "weight",
            torch.ones(image_channels).index_fill(0, torch.arange(153, 158), 20),
        )

    def forward(self, qloss, target, prediction):
        bce_loss = F.binary_cross_entropy_with_logits(
            prediction.permute(0, 2, 3, 1),
            target.permute(0, 2, 3, 1),
            pos_weight=self.weight,
        )
        loss = bce_loss + self.codebook_weight * qloss
        return loss


class VQVAEWithBCELoss(nn.Module):
    def __init__(self, image_channels=159, codebook_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.register_buffer(
            "weight",
            torch.ones(image_channels).index_fill(0, torch.arange(153, 158), 20),
        )

    def forward(self, qloss, target, prediction):
        bce_mse_loss = F.mse_loss(prediction.sigmoid(), target) + F.binary_cross_entropy_with_logits(
            prediction.permute(0, 2, 3, 1),
            target.permute(0, 2, 3, 1),
            pos_weight=self.weight,
        )
        loss = bce_mse_loss + self.codebook_weight * qloss
        return loss
