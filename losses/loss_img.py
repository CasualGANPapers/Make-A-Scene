import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from discriminator import Discriminator, weights_init
from face_loss import resnet50
from torchvision.transforms.functional import crop


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
            torch.mean(torch.nn.functional.softplus(-logits_real)) +
            torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_factor=1.0, disc_weight=1.0, perceptual_weight=1.0, disc_conditional=False):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.face_loss = resnet50()

        self.discriminator = Discriminator().apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, bbox_face, bbox_obj, global_step, last_layer=None, cond=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
        rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        face_loss = 0.
        for img, rec, bboxes in zip(inputs, reconstructions, bbox_face):
            for bbox in bboxes:
                img = crop(img, *bbox)  # bbox needs to be [x, y, height, width]
                rec = crop(rec, *bbox)
                face_loss += self.face_loss(img.contiguous(), rec.contiguous())

        object_loss = 0.
        for img, rec, bboxes in zip(inputs, reconstructions, bbox_obj):
            for bbox in bboxes:
                img = crop(img, *bbox)  # bbox needs to be [x, y, height, width]
                rec = crop(rec, *bbox)
                object_loss += self.object_loss(img.contiguous(), rec.contiguous())

        # now the GAN part
        if cond is None:
            assert not self.disc_conditional
            logits_fake = self.discriminator(reconstructions.contiguous())
        else:
            assert self.disc_conditional
            logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
        g_loss = -torch.mean(logits_fake)

        d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)

        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

        loss = nll_loss + \
               d_weight * disc_factor * g_loss + \
               self.codebook_weight * codebook_loss.mean() + \
               face_loss + \
               object_loss

        if cond is None:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
        else:
            logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
            logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

        d_loss = disc_factor * hinge_d_loss(logits_real, logits_fake)

        return loss, d_loss