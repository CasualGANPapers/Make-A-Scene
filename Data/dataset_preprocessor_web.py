import bisect
import numpy as np
import os
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset as ConcatDataset_
from tqdm import tqdm
import torch.multiprocessing as mp
import warnings
from torchvision import transforms
from hydra.utils import instantiate
import albumentations as A
from albumentations.pytorch import ToTensorV2
from urllib.request import urlretrieve
from webdataset import WebDataset
from webdataset.shardlists import split_by_node

class PreprocessData:
    def __init__(self, ready_queue):
        self.transforms = A.Compose([
            A.SmallestMaxSize(512),
            A.CenterCrop(512, 512, always_apply=True),
            ToTensorV2(transpose_mask=True)
        ])
        self.lasttar = "no"
        self.ready_queue = ready_queue
    def __call__(self, data):
        result = self.transforms(image=data["jpg"])
        data["jpg"] = result["image"]
        data["tarname"] = os.path.basename(data["__url__"])
        if self.lasttar!=data["tarname"]:
            rank = os.environ["RANK"]
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker = worker_info.id
            else:
                worker = None

            if self.lasttar != "no":
                self.ready_queue.put("%s/%s/processed/%s" % (rank, worker, self.lasttar))
                self.ready_queue.put("%s/%s/started/%s" % (rank, worker, data["tarname"]))
                print(self.lasttar, "processed!")
            self.lasttar = data["tarname"]
        return data

class UnprocessedWebDataset(WebDataset):
    def __init__(self, root, *args, ready_queue=None, **kwargs):
        shards = [os.path.join(root, filename) for filename in os.listdir(root) if os.path.splitext(filename)[1]==".tar"]
        super().__init__(shards, *args, nodesplitter=split_by_node, **kwargs)
        self.decode("rgb")
        self.map(PreprocessData(ready_queue))
        self.to_tuple("__key__", "tarname", "jpg")

class ProcessData:
    def __init__(self,):
        self.transforms = A.Compose([
            A.SmallestMaxSize(256),
            A.RandomCrop(256, 256, always_apply=True),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=True)
        ], bbox_params=A.BboxParams(format='pascal_voc', min_area=100, min_visibility=0.2),
            additional_targets={"bboxes0": "bboxes"})

    def __call__(self, data):
        npz_data = data["npz"]
        # Panoptic
        seg_panoptic = F.one_hot(
            torch.from_numpy(npz_data["seg_panoptic"] + 1).to(torch.long), num_classes=134
        )[..., 1:]
        edges_panoptic = torch.from_numpy(npz_data["edge_panoptic"]).unsqueeze(-1)
        box_thing = npz_data["box_things"].tolist()
        for box in box_thing:
            box.append(0)

        # Human parts
        seg_human = F.one_hot(
            torch.from_numpy(npz_data["seg_human"] + 1).to(torch.long), num_classes=21
        )[..., 1:]
        edges_human = torch.from_numpy(npz_data["edge_human"]).unsqueeze(-1)

        # Edges
        seg_edges = (edges_panoptic + edges_human).float()

        # Face
        seg_face = F.one_hot(
            torch.from_numpy(npz_data["seg_face"]).to(torch.long), num_classes=6
        )[..., 1:]
        box_face = npz_data["box_face"].tolist()
        for box in box_face:
            box.append(0)

        # Concatenate masks
        seg_map = torch.cat(
            [seg_panoptic, seg_human, seg_face, seg_edges], dim=-1
        ).numpy()

        transformed_data = self.transforms(image=data["jpg"], bboxes=box_thing, bboxes0=box_face,)
        data["jpg"] = transformed_data["image"]
        data["mask"] = seg_map
        data["box_things"] = transformed_data["bboxes"]
        data["box_face"] = transformed_data["bboxes0"]
        return data

class PreprocessedWebDataset(WebDataset):
    def __init__(self, root, *args, **kwargs):
        shards = [os.path.join(root, filename) for filename in os.listdir(root) if os.path.splitext(filename)[1]==".tar"]
        super().__init__(shards, *args, nodesplitter=split_by_node, **kwargs)
        self.decode("rgb")
        #self.decode("npz")
        self.map(ProcessData())
        self.to_tuple("jpg", "mask", "box_things", "box_face", "txt")


if __name__ == "__main__":
    coco = COCO2014Dataset(
        "./mydb", "./mydb/preprocessed"
    )
    from torchvision.utils import draw_bounding_boxes
    import matplotlib.pyplot as plt

    img, _, ft, fb, _ = coco[0]
    plt.imshow(draw_bounding_boxes(img, torch.tensor(ft + fb)).permute(1, 2, 0))
    plt.show()
    print()
