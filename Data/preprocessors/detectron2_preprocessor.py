import os
import cv2
import torch
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes
import numpy as np
import detectron2
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from .edge_extractor import get_edges


class PanopticPreprocesor:
    proc_type = "panoptic"
    def __init__(
        self,
        config="/home/silent/hdd/nets/detectron2/projects/Panoptic-DeepLab/configs/COCO-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml",
        num_classes=133,
        model_weights="/home/silent/hdd/nets/db/model_final_5e6da2.pkl",
        device=None,
    ):
        self.num_classes = num_classes

        cfg = get_cfg()
        print(cfg.MODEL.DEVICE)
        opts = ["MODEL.WEIGHTS", model_weights]
        add_panoptic_deeplab_config(cfg)
        cfg.merge_from_file(config)
        cfg.merge_from_list(opts)
        if device is not None:
            cfg.MODEL.DEVICE = device
        cfg.freeze()
        print("Building model")
        self.predictor = DefaultPredictor(cfg)


    def bounding_boxes(self, panoptic):
        obj_ids = torch.unique(panoptic)
        thing_ids = obj_ids[obj_ids/1000 < 80] # according to panopticapi first 80 classes are "things"
        binary_masks = panoptic == thing_ids[:, None, None]
        boxes = masks_to_boxes(binary_masks)
        return boxes

    def __call__(self, img):
        data = {}
        # Returns tensor of shape [H, W] with values equal to 1000*class_id + instance_idx
        panoptic = self.predictor(img)["panoptic_seg"][0].cpu()
        bounding_boxes = self.bounding_boxes(panoptic)
        edges = get_edges(panoptic.numpy())
        data["seg_panoptic"] = np.array(panoptic // 1000, dtype=np.uint8)
        data["box_things"] = bounding_boxes
        data["edges"] =  edges.astype(bool)
        return data


if __name__ == "__main__":
    img = cv2.imread("test.png")
    detectron2 = PanopticPreprocesor()
    panoptic = detectron2(img)
    print(panoptic)
    torch.save(panoptic, "test.pth")
