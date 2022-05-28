import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import os


class Logger:
    def __init__(self, proc_id, log_dir=".", device="cuda"):
        self.proc_id = proc_id
        if proc_id != 0:
            return
        self.writer = SummaryWriter(log_dir)
        self.step = 0
        os.makedirs("./results")

    def log(self, step=None, img=None, img_rec=None, **kwargs):
        if self.proc_id != 0:
            return
        self.step = step if step is not None else self.step + 1
        for key in kwargs:
            self.writer.add_scalar(key, kwargs[key].detach().cpu().item(), self.step)
        if img is not None and img_rec is not None and self.step%500==0:
            img = img.detach().cpu()
            img_rec = img_rec.detach().cpu()
            pairs = torch.cat([img,img_rec]).detach().cpu()
            img_grid = vutils.make_grid(pairs)
            self.writer.add_image('samples', img_grid.detach().cpu(), global_step=step)


class Visualizer:
    dims = {
        "panoptic": [0, 133],
        "human": [133, 153],
        "face": [153, 158],
        "edge": [158, 159]
    }

    def __init__(self, log_dir=".", device="cuda"):
        self.weights = {}
        for key in self.dims:
            size = self.dims[key][1] - self.dims[key][0]
            weight = torch.randn([3, size, 1, 1]).to(device)
            self.weights[key] = weight
        os.makedirs("./results")

    def log_images(self, seg, seg_rec):
        seg = self.colorize(seg)
        seg_rec = self.colorize(seg_rec, logits=True)
        both = torch.cat((seg, seg_rec))
        grid = vutils.make_grid(both, nrow=2)
        vutils.save_image(both, f"./results/{self.step}.jpg", nrow=4)

    def colorize(self, seg, logits=False):
        results = []
        for key in self.dims:
            seg_key = seg[:, self.dims[key][0]: self.dims[key][1]]
            if logits:
                n_classes = seg_key.shape[1]
                if "face" in key or "edge" in key:
                    mask = seg_key.sigmoid() > 0.2
                seg_key = torch.argmax(seg_key, dim=1, keepdim=False)
                seg_key = F.one_hot(seg_key, num_classes=n_classes)
                seg_key = seg_key.permute(0, 3, 1, 2).float()
                if "face" in key or "edge" in key:
                    seg_key *= mask

            weight = self.weights[key]
            with torch.no_grad():
                x = F.conv2d(seg_key, weight)
                x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
                results.append(x)
        return results
        seg = seg[:, :133]
        if logits:
            #    seg = (seg.sigmoid()>0.35).to(torch.float)
            seg = torch.argmax(seg, dim=1, keepdim=False)
            seg = F.one_hot(seg, num_classes=133)
            seg = seg.squeeze(1).permute(0, 3, 1, 2).float()

        return x

    def __call__(self, step, image=None, seg=None, seg_rec=None):
        results = [image]
        if seg is not None:
            results.extend(self.colorize(seg))
        results.append(torch.zeros_like(image))
        if seg_rec is not None:
            results.extend(self.colorize(seg_rec, logits=True))
        results = torch.cat(results)
        vutils.save_image(results, f"./results/result_{step}.jpg", nrow=5)
