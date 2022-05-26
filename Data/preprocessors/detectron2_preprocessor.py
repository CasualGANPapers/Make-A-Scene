import os
import cv2
import torch
import torch.nn.functional as F
# from torchvision.ops import masks_to_boxes
import numpy as np
import detectron2
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from .edge_extractor import get_edges


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)
    n = masks.shape[0]
    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)
    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes


class Predictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.input_format = cfg.INPUT.FORMAT

    def __call__(self, imgs: np.array):
        # imgs should be numpy b x c x h x w
        with torch.no_grad():
            #imgs = torch.as_tensor(imgs.astype("float32"))
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                imgs = imgs.flip([1])
            
            height, width = imgs.shape[2:]

            inputs = [{"image": image, "height": height, "width": width} for image in imgs]
            predictions = self.model(inputs)
            return predictions


class PanopticPreprocesor:
    proc_type = "panoptic"
    def __init__(
        self,
        config="/home/ubuntu/anaconda3/envs/schp/lib/python3.8/site-packages/detectron2/projects/panoptic_deeplab/configs/COCO-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml",
        num_classes=133,
        model_weights="/home/ubuntu/anaconda3/envs/schp/lib/python3.8/site-packages/detectron2/projects/panoptic_deeplab/configs/COCO-PanopticSegmentation/model_final_5e6da2.pkl",
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
        self.predictor = Predictor(cfg)

    def bounding_boxes(self, panoptics):
        all_boxes = []
        for panoptic in panoptics:
            #panoptic = torch.Tensor(panoptic)
            obj_ids = torch.unique(panoptic)
            thing_ids = obj_ids[obj_ids/1000 < 80] # according to panopticapi first 80 classes are "things"
            binary_masks = panoptic == thing_ids[:, None, None]
            boxes = masks_to_boxes(binary_masks)
            all_boxes.append(boxes.cpu().numpy())
        return all_boxes

    def __call__(self, imgs: np.array):
        # imgs should be numpy b x c x h x w
        data = {}
        # Returns tensor of shape [H, W] with values equal to 1000*class_id + instance_idx
        # panoptic = self.predictor(imgs)["panoptic_seg"][0].cpu()
        panoptic = self.predictor(imgs)
        panoptic = list(map(lambda pan: pan["panoptic_seg"][0], panoptic))
        bounding_boxes = self.bounding_boxes(panoptic)
        panoptic = list(map(lambda pan: pan.cpu().numpy(), panoptic))
        panoptic = np.array(panoptic)
        edges = get_edges(panoptic)
        data["seg_panoptic"] = np.array(panoptic // 1000, dtype=np.uint8)
        data["box_things"] = bounding_boxes
        data["edges"] = edges.astype(bool)
        return data


if __name__ == "__main__":
    # img = cv2.imread("humans.jpg")
    img = np.random.randint(0, 255, (5, 3, 300, 300), dtype=np.uint8)
    detectron2 = PanopticPreprocesor()
    panoptic = detectron2(img)
    print(panoptic)
    # torch.save(panoptic, "test.pth")
