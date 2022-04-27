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

try:
    from .preprocessors import Detectron2, HumanParts, HumanFace, get_edges
except ModuleNotFoundError as e:
    #print("Missing some dependecies, required for data preprocessing:", e)
    pass


class PreprocessedDataset(Dataset):
    def __init__(
        self,
        root=None,
        image_dirs=None,
        preprocessed_folder=None,
    ):
        self.image_dirs = image_dirs

        self.preprocessed_folder = preprocessed_folder
        self.preprocessed_path = os.path.join( preprocessed_folder, "segmentations", "%s_%s.npz",)

        self.root = root
        self.transforms = A.Compose([
                                    A.SmallestMaxSize(256),
                                    A.RandomCrop(256, 256, always_apply=True),
                                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                    ToTensorV2(transpose_mask=True)
                                    ])

        if not os.path.exists(preprocessed_folder):
            os.makedirs(preprocessed_folder)
        else:
            assert os.path.isdir(preprocessed_folder)

        self.img_names_path = os.path.join(preprocessed_folder, f"img_names_{self.name}.npz")
        if not os.path.exists(self.img_names_path):
            self.parse_image_names()

        img_list = np.load(self.img_names_path)
        self.img_names = img_list["img_names"]
        if "img_urls" in img_list:
            self.img_urls = img_list["img_urls"]

    def load_segmentation(self, idx):
        img_name = os.path.splitext(self.img_names[idx])[0]

        data_panoptic = np.load(self.preprocessed_path % (img_name, "panoptic"))
        data_human = np.load(self.preprocessed_path % (img_name, "human"))
        data_face = np.load(self.preprocessed_path % (img_name, "face"))

        # Panoptic
        seg_panoptic = F.one_hot(
            torch.from_numpy(data_panoptic["seg_panoptic"] + 1).to(torch.long), num_classes=134
        )[..., 1:]
        edges_panoptic = torch.from_numpy(data_panoptic["edges"]).unsqueeze(-1)
        box_thing = data_panoptic["box_things"]

        # Human parts
        seg_human = F.one_hot(
            torch.from_numpy(data_human["seg_human"] + 1).to(torch.long), num_classes=21
        )[..., 1:]
        edges_human = torch.from_numpy(data_human["edges"]).unsqueeze(-1)

        # Edges
        seg_edges = (edges_panoptic + edges_human).float()

        #Face
        seg_face = F.one_hot(
            torch.from_numpy(data_face["seg_face"]).to(torch.long), num_classes=6
        )[..., 1:]
        box_face = data_face["box_face"]

        # Concatenate masks
        seg_map = torch.cat(
            [seg_panoptic, seg_human, seg_face, seg_edges], dim=-1
        )

        return np.array(seg_map), box_thing, box_face

    def __getitem__(self, idx):
        segmentation, box_thing, box_face = self.load_segmentation(idx)
        image, _ = self.get_image(idx)
        data = self.transforms(image=image, mask=segmentation, bboxes=box_thing, bboxes0=box_face)
        return data["image"], data["mask"], data["bboxes"], data["bboxes0"], self.img_names[idx]

    def get_image(self, idx):
        raise NotImplementedError

    def parse_image_names(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.img_names)


class BaseCOCODataset(PreprocessedDataset):
    def get_image(self, idx):
        img_name = self.img_names[idx]
        path = os.path.join(self.root, img_name)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, img_name

    def parse_image_names(self):
        img_names = []
        for directory in self.image_dirs:
            for filename in os.listdir(os.path.join(self.root, directory)):
                if os.path.splitext(filename)[1] in [".jpg", ".png"]:
                    img_names.append(os.path.join(directory, filename))
        np.savez(self.img_names_path, img_names=img_names)


class COCO2014Dataset(BaseCOCODataset):
    name = "coco2014"
    image_dirs = "train2014"
    def __init__(self, root, preprocessed_folder, **kwargs):
        super().__init__(
            root=root,
            image_dirs=["train2014"],
            preprocessed_folder=preprocessed_folder,
            **kwargs,
        )


class COCO2017Dataset(BaseCOCODataset):
    name = "coco2017"
    image_dirs = "train2017"
    def __init__(self, root, preprocessed_folder, **kwargs):
        super().__init__(
            root=root,
            image_dirs=["train2017"],
            preprocessed_folder=preprocessed_folder,
            **kwargs,
        )


class Conceptual12mDataset(PreprocessedDataset):
    name = "cc12m"
    def __init__(self, root, preprocessed_folder, **kwargs):
        super().__init__(
            root=root,
            **kwargs,
        )

    def parse_image_names(self, listfile):
        img_names = []
        img_urls = []
        with open(listfile, "r") as urllist:
            for i, line in enumerate(urllist):
                url, caption = line.split("\t")
                caption = caption.strip()
                img_names.append(caption+".jpg")
        np.savez(self.img_names_path, img_names=img_names, img_urls=img_urls)



    def get_image(self, idx):
        img_name = self.img_names[idx]
        path = os.path.join(self.root, img_name)
        if not os.path.exists(path):
            self.download_image(self.url[idx], img_name)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, img_name



    def download_image(self, url, image_name):
        try:
            image_path = os.path.join(self.root, image_name)
            urlretrieve(url, image_path)
            return True
        except HTTPError:
            print("Failed to download the image: ", image_name)
            return False


class ConcatDataset(ConcatDataset_):
    def get_true_idx(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_image(self, idx):
        dataset_idx, sample_idx = self.get_true_idx(idx)
        return self.datasets[dataset_idx].get_image(sample_idx)


if __name__ == "__main__":
    coco = COCO2014Dataset(
        "/path_to_coco", "/path_to_preprocessed_folder", proc_total=8
    )
    print(coco[0])
