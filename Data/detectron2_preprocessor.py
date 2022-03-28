import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import detectron2
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

class PanopticPreprocesor:
    def __init__(
        self,
        one_channel=True,
        config="../../detectron2/projects/Panoptic-DeepLab/configs/COCO-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml",
        num_classes=133,
        model_weights="model_final_5e6da2.pkl",
    ):
        self.num_classes = num_classes
        self.one_channel = one_channel

        cfg = get_cfg()
        opts = ["MODEL.WEIGHTS", model_weights]
        add_panoptic_deeplab_config(cfg)
        cfg.merge_from_file(config)
        cfg.merge_from_list(opts)
        cfg.freeze()
        print("Building model")
        self.predictor = DefaultPredictor(cfg)

    def panoptic_seg_one_channel(self, img):
        # Returns tensor of shape [H, W] with values equal to 1000*class_id + instance_idx
        panoptic = self.predictor(img)["panoptic_seg"][0]
        return panoptic

    def semantic_segment_one_hot(self, img):
        # Returns tensor of shape [H, W, C] with binary masks of classes.
        panoptic = self.panoptic_seg_one_channel(img)
        return F.one_hot(panoptic//1000, num_classes=self.num_classes)

    def __call__(self, img):
        if self.one_channel:
            return self.panoptic_seg_one_channel(img)
        else:
            return self.semantic_segment_one_hot(img)



if __name__ == "__main__":
    img = cv2.imread("test.jpg")
    detectron2 = PanopticPreprocesor(one_channel=False)
    panoptic = detectron2(img)
    print(panoptic)
    torch.save(panoptic, "test.pth")

