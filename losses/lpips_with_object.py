import os
import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision.models import vgg16
from collections import namedtuple
import requests
from tqdm import tqdm
from .lpips import LPIPS


class WeightThingGrad(Function):
    def forward(ctx, input, bboxes):
        weights = torch.ones_like(input)
        for weight, img_bboxes in zip(weights, bboxes):
            for x_min, y_min, x_max, y_max in img_bboxes:
                weight[:, x_min:x_max, y_min:y_max]
                
        ctx.save_for_backward(weights)
        return input

    def backward(ctx, grad_output):
        weights = ctx.saved_tensors[0]
        return grad_output*weights, None

weight_thing_grad = WeightThingGrad.apply

class LPIPSWithObject(LPIPS):
    def forward(self, real_x, fake_x, object_boxes):
        fake_x = weight_thing_grad(fake_x, object_boxes)
        return super().forward(real_x, fake_x)

