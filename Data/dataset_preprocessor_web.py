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
from .utils import check_bboxes
from urllib.request import urlretrieve
from webdataset import WebDataset
from webdataset.shardlists import split_by_node
from webdataset.handlers import warn_and_continue
from itertools import islice

def my_split_by_node(src, group=None):
    rank, world_size, = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world_size > 1:
        for s in islice(islice(src, (rank*2)//world_size, None, 2), rank%(world_size//2), None, world_size//2):
            yield s
    else:
        for s in src:
            yield s


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
            proc_type = os.environ["TYPE"]
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker = worker_info.id
            else:
                worker = None

            if self.lasttar != "no":
                self.ready_queue.put("%s/%s/processed/%s" % (rank+"-"+proc_type, worker, self.lasttar))
                print(self.lasttar, "processed!")
            self.ready_queue.put("%s/%s/started/%s" % (rank+"-"+proc_type, worker, data["tarname"]))
            self.lasttar = data["tarname"]
        return data

class UnprocessedWebDataset(WebDataset):
    def __init__(self, root, *args, is_dir=False, ready_queue=None, **kwargs):
        if is_dir:
            shards = [os.path.join(root, filename) for filename in os.listdir(root) if os.path.splitext(filename)[1]==".tar"]
            shards.sort()
            self.basedir = root
        else:
            shards = root
            self.basedir = os.path.dirname(root)
        super().__init__(shards, *args, nodesplitter=my_split_by_node, handler=warn_and_continue, **kwargs)
        self.decode("rgb")
        self.map(PreprocessData(ready_queue))
        self.to_tuple("__key__", "tarname", "jpg")


class ProcessData:
    def __init__(self,):
        self.pretransforms = A.Compose([
            A.SmallestMaxSize(512),
            A.CenterCrop(512, 512, always_apply=True),
        ])
        self.transforms = A.Compose([
            #A.SmallestMaxSize(512),
            #A.CenterCrop(512, 512, always_apply=True),
            #A.RandomCrop(256, 256, always_apply=True),
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

        data["jpg"] = self.pretransforms(image=data["jpg"])["image"]
        box_thing = check_bboxes(box_thing)
        box_face = check_bboxes(box_face)
        transformed_data = self.transforms(image=data["jpg"], bboxes=box_thing, bboxes0=box_face,)
        data["jpg"] = transformed_data["image"]
        data["mask"] = seg_map
        data["box_things"] = transformed_data["bboxes"]
        data["box_face"] = transformed_data["bboxes0"]
        return data


class PreprocessedWebDataset(WebDataset):
    def __init__(self, url, *args, **kwargs):
        super().__init__(url, *args, nodesplitter=split_by_node, handler=warn_and_continue, **kwargs)
        self.decode("rgb")
        #self.decode("npz")
        self.map(ProcessData(), handler=warn_and_continue)
        self.to_tuple("jpg", "mask", "box_things", "box_face", "txt", handler=warn_and_continue)

class COCOWebDataset(PreprocessedWebDataset):
    def __init__(self, *args, **kwargs):
        super().__init__("pipe:aws s3 cp s3://s-mas/coco_processed/{00000..00010}.tar -", *args, **kwargs)

class CC3MWebDataset(PreprocessedWebDataset):
    def __init__(self, *args, **kwargs):
        super().__init__("pipe:aws s3 cp s3://s-mas/cc3m_processed/{00000..00311}.tar -", *args, **kwargs)

class S3ProcessedDataset(PreprocessedWebDataset):
    datasets = {
                "coco": "pipe:aws s3 cp s3://s-mas/coco_processed/{00000..00059}.tar -",
                "cc3m": "pipe:aws s3 cp s3://s-mas/cc3m_processed/{00000..00331}.tar -",
                "cc12m": "pipe:aws s3 cp s3://s-mas/cc12m_processed/{00000..01242}.tar -",
                "laion": "pipe:aws s3 cp s3://s-mas/laion_en_processed/{00000..01500}.tar -"
                }
    def __init__(self, names, *args, **kwargs):
        urls = []
        for name in names:
            assert name in self.datasets, f"There is no processed dataset {name}"
            urls.append(self.datasets[name])
        urls = "::".join(urls)
        super().__init__(urls, *args, **kwargs)

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

