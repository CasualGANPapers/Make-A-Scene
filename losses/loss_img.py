import torch
import torch.nn as nn
import torch.nn.functional as F
from .lpips import LPIPS
from .lpips_with_object import LPIPSWithObject
from .discriminator import Discriminator, weights_init
from .face_loss import FaceLoss
from torchvision.transforms.functional import crop


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
    ):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPSWithObject().eval()
        self.perceptual_weight = perceptual_weight

        self.face_loss = FaceLoss()
        #self.object_loss = self.perceptual_loss

        self.discriminator = Discriminator().apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer.weight, retain_graph=True)[
            0
        ]
        g_grads = torch.autograd.grad(g_loss, last_layer.weight, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        optimizer_idx,
        global_step,
        images,
        reconstructions,
        codebook_loss=None,
        bbox_obj=None,
        bbox_face=None,
        last_layer=None,
    ):
        if optimizer_idx == 0:  # vqvae loss
            rec_loss = torch.abs(images.contiguous() - reconstructions.contiguous())
            p_loss = self.perceptual_loss(
                images.contiguous(), reconstructions.contiguous(), bbox_obj
            )
            rec_loss = rec_loss + self.perceptual_weight * p_loss

            nll_loss = rec_loss
            nll_loss = torch.mean(nll_loss)

            face_loss = self.face_loss(images, reconstructions, bbox_face)

            object_loss = images.new_tensor(0)
            #for img, rec, bboxes in zip(images, reconstructions, bbox_obj):
            #    img_object_loss = img.new_tensor(0)
            #    for bbox in bboxes:
            #        # xmin, ymin, xmax, ymax
            #        top = bbox[1]
            #        left = bbox[0]
            #        height = bbox[3] - bbox[1]
            #        width = bbox[2] - bbox[0]
            #        crop_img = crop(img, top, left, height, width).unsqueeze(
            #            0
            #        )  # bbox needs to be [x, y, height, width]
            #        crop_rec = crop(rec, top, left, height, width).unsqueeze(0)
            #        img_object_loss += self.object_loss(
            #            crop_img.contiguous(), crop_rec.contiguous()
            #        ).mean()  # TODO: check if crops are actually correct
            #    object_loss += img_object_loss/(len(bboxes)+1)

            logits_fake = self.discriminator(
                reconstructions.contiguous()
            )  # cont not necessary
            g_loss = -torch.mean(logits_fake)

            d_weight = self.calculate_adaptive_weight(
                nll_loss, g_loss, last_layer=last_layer
            )

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            loss = (
                nll_loss
                + d_weight * disc_factor * g_loss
                + self.codebook_weight * codebook_loss.mean()
                + face_loss
                + object_loss
            )
            # 0.001 * face_loss
            return loss, (nll_loss, object_loss, face_loss)
            # return loss, (nll_loss, images.new_tensor(0), images.new_tensor(0))

        if optimizer_idx == 1:  # gan loss
            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            logits_real = self.discriminator(images.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            d_loss = disc_factor * hinge_d_loss(logits_real, logits_fake)
            return d_loss
