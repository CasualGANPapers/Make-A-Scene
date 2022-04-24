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
        name
        root=None,
        image_dirs=None,
        preprocessed_folder=None,
        force_preprocessing=True,
        detectron2=None,
        human_parsing=None,
        human_face=None,
        proc_total=1,
    ):
        self.name = name
        self.image_dirs = image_dirs

        # Preprocessing modules, required only during preprocessing or online-processing
        self.detectron2 = detectron2
        self.human_parsing = human_parsing
        self.human_face = human_face

        self.preprocessed_folder = preprocessed_folder
        self.root = root
        self.proc_total = proc_total
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

        self.img_names_path = os.path.join(preprocessed_folder, f"img_names_{name}.npz")
        if not os.path.exists(self.img_names_path) or force_preprocessing:
            self.parse_image_names()

        img_list = np.load(self.img_names_path)
        self.img_names = img_list["img_names"]
        if "img_urls" in img_list:
            self.img_urls = img_list["img_urls"]

        if not os.path.exists(preprocessed_folder) or force_preprocessing:
            self.preprocess_dataset()

    def preprocess_dataset(self):
        assert torch.cuda.is_available(), "GPU required for preprocessing"
        if self.proc_total > torch.cuda.device_count():
            warnings.warn(
                "Do not recommended to set more processes, than you have GPUs"
            )
        if self.proc_total == 1:
            if self.detectron2 is None:
                self.detectron2 = Detectron2()
            if self.human_parsing is None:
                self.human_parsing = HumanParts()
            if self.human_face is None:
                self.human_face = HumanFace()
            for img_name in tqdm(self.img_names):
                self.preprocess_single_image(img_name)
        else:
            if self.detectron2 is None:
                self.detectron2 = [None] * self.proc_total
            if self.human_parsing is None:
                self.human_parsing = [None] * self.proc_total
            if self.human_face is None:
                self.human_face = [None] * self.proc_total
            procs = []
            mp.set_start_method("spawn")
            for proc_id in range(self.proc_total):
                p = mp.Process(target=self.preprocess_single_process, args=(proc_id,))
                p.start()
                procs.append(p)
            for proc in procs:
                proc.join()
            self.img_names = []
            for proc_id in range(self.proc_total):
                names = np.load(f"tmp_names_{self.name}_{proc_id}.npz")["names"]
                os.remove(f"tmp_names_{self.name}_{proc_id}.npz")
                self.img_names.extend(names)
            print(len(self.img_names))
            np.savez( self.img_names_path, img_names=self.img_names)

    def preprocess_single_process(self, proc_id,):
        correct_names = []
        device = f"cuda:{proc_id%torch.cuda.device_count()}"
        torch.cuda.set_device(  # https://github.com/pytorch/pytorch/issues/21819#issuecomment-553310128
            proc_id % self.proc_total
        )
        if self.detectron2[proc_id] is None:
            self.detectron2[proc_id] = Detectron2(device=device)
        if self.human_parsing[proc_id] is None:
            self.human_parsing[proc_id] = HumanParts(device=device)
        if self.human_face[proc_id] is None:
            self.human_face[proc_id] = HumanFace(device=device)
        img_names = self.img_names[proc_id :: self.proc_total]
        bar = tqdm(img_names) if proc_id == 0 else img_names
        for img_name in bar:
            if self.preprocess_single_image(img_name, proc_id,):
                correct_names.append(img_name)
        np.savez(f"tmp_names_{proc_id}", names=correct_names)

    def preprocess_single_image(self, img_name, proc_id=None, correct_img_names=None):
        image = cv2.imread(os.path.join(self.root, img_name))
        # Detectron2 panoptic segmentation
        detectron2 = self.detectron2 if proc_id is None else self.detectron2[proc_id]
        if detectron2 is not None:
            seg_panoptic, box_thing = detectron2(image).to(torch.int32).cpu().numpy()
            seg_edges = get_edges(seg_panoptic)
            seg_panoptic = np.array(seg_panoptic // 1000, dtype=np.uint8)
            seg_edges = np.array(seg_edges, dtype=np.bool)
            save_path = os.path.join(
                self.preprocessed_folder, "segmentations", os.path.splitext(img_name)[0] + "_panoptic"
            )
            np.savez_compressed(
                save_path,
                seg_panoptic=seg_panoptic,
                box_thing=box_thing,
                edges=seg_edges,
            )

        # Human parts
        human_parsing = (
            self.human_parsing if proc_id is None else self.human_parsing[proc_id]
        )
        if human_parsing is not None:
            seg_human = human_parsing(image).to(torch.int32).cpu().numpy()
            seg_edges = get_edges(seg_human)
            seg_human = np.array(seg_human, dtype=np.uint8)
            seg_edges = np.array(seg_edges, dtype=np.bool)
            save_path = os.path.join(
                self.preprocessed_folder, "segmentations", os.path.splitext(img_name)[0] + "_human"
            )
            np.savez_compressed(
                save_path,
                seg_human=seg_human,
                edges=seg_edges,
            )

        # Human face alignment
        human_face = self.human_face if proc_id is None else self.human_face[proc_id]
        if human_face is not None:
            seg_face, box_face = human_face(image)
            seg_face = np.array(seg_face, dtype=np.uint8)
            save_path = os.path.join(
                self.preprocessed_folder, "segmentations", os.path.splitext(img_name)[0] + "_face"
            )
            np.savez_compressed(
                save_path,
                seg_face=seg_face,
                box_face=box_face,
            )
        return True

    def load_segmentation(self, img_name):
        # Panoptic
        data_panoptic = np.load(
            os.path.join(
                self.preprocessed_folder,
                "segmentations",
                os.path.splitext(img_name)[0] + "_panoptic.npz",
            )
        )
        seg_panoptic = F.one_hot(
            torch.from_numpy(data_panoptic["seg_panoptic"] + 1).to(torch.long), num_classes=134
        )[..., 1:]
        # Human parts
        data_human = np.load(
            os.path.join(
                self.preprocessed_folder,
                "segmentations",
                os.path.splitext(img_name)[0] + "_human.npz",
            )
        )
        seg_human = F.one_hot(
            torch.from_numpy(data_human["seg_human"] + 1).to(torch.long), num_classes=21
        )[..., 1:]

        #Face
        data_face = np.load(
            os.path.join(
                self.preprocessed_folder,
                "segmentations",
                os.path.splitext(img_name)[0] + "_face.npz",
            )
        )
        seg_face = F.one_hot(
            torch.from_numpy(data_face["seg_face"]).to(torch.long), num_classes=6
        )[..., 1:]

        edges_panoptic = torch.from_numpy(data_panoptic["edges"]).unsqueeze(-1)
        edges_human = torch.from_numpy(data_human["edges"]).unsqueeze(-1)
        seg_edges = ((edges_panoptic + edges_human)>0).float()

        seg_map = torch.cat(
            [seg_panoptic, seg_human, seg_face, seg_edges], dim=-1
        )#.permute(2, 0, 1)
        return seg_map, data_panoptic["box_thing"], data_face["box_face"]

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        try:
            segmentation = self.load_segmentation(img_name)
        except FileNotFoundError:
            self.preprocess_single_image(img_name)
            segmentation, box_thing, box_face = self.load_segmentation(img_name)
        image = self.get_image(idx)
        data = self.transforms(image=image, mask=segmentation.numpy(), bboxes=box_thing, bboxes0=box_face)
        return data["image"], data["mask"].float(), data["bboxes"], data["bboxes0"]

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
        return image

    def parse_image_names(self):
        img_names = []
        for directory in self.img_dirs:
            for filename in os.listdir(os.path.join(self.root, directory)):
                if os.path.splitext(filename)[1] in [".jpg", ".png"]:
                    img_names.append(os.path.join(directory, filename))
        np.savez(self.img_names_path, img_names=img_names)


class COCO2014Dataset(BaseCOCODataset):
    def __init__(self, root, preprocessed_folder, **kwargs):
        super().__init__(
            "coco2014"
            root=root,
            image_dirs=["train2014"],
            preprocessed_folder=preprocessed_folder,
            **kwargs,
        )


class COCO2017Dataset(BaseCOCODataset):
    def __init__(self, root, preprocessed_folder, **kwargs):
        super().__init__(
            "coco2017"
            root=root,
            image_dirs=["train2017"],
            preprocessed_folder=preprocessed_folder,
            **kwargs,
        )


class Conceptual12mDataset(PreprocessedDataset):
    def __init__(self, root, preprocessed_folder, **kwargs):
        super().__init__(
            "cc12m"
            root=root,
            image_dirs=["cc12m_images"],
            preprocessed_folder=preprocessed_folder,
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
        return image



    def download_image(self, url, image_name):
        try:
            image_path = os.path.join(self.root, image_name)
            urlretrieve(url, image_path)
            return True
        except HTTPError:
            print("Failed to download the image: ", image_name)
            return False


class ConcatDataset(ConcatDataset_):
    def __init__(self, datasets_configs, proc_total=1, force_preprocessing=False, ):
        datasets = []
        self.detectron2 = None
        self.human_parsing = None
        self.human_face = None
        if force_preprocessing:
            if proc_total == 1:
                self.detectron2 = Detectron2()
                self.human_parsing = HumanParts()
                self.human_face = HumanFace()
        else:
                self.detectron2 = [None] * self.proc_total
                self.human_parsing = [None] * self.proc_total
                self.human_face = [None] * self.proc_total
                for proc_id in range(proc_total):
                    device = f"cuda:{proc_id%torch.cuda.device_count()}"
                    self.detectron2[proc_id] = Detectron2(device=device)
                    self.human_parsing[proc_id] = HumanParts(device=device)
                    self.human_face[proc_id] = HumanFace(device=device)


        for config in datasets_configs:
            dataset = instantiate(config,
                                  force_preprocessing=force_preprocessing,
                                  proc_total=proc_total,
                                  detectron2=self.detectron2,
                                  human_face=self.human_face,
                                  human_parsing=self.human_parsing)
            datasets.append(dataset)
        super().__init__(datasets)


if __name__ == "__main__":
    coco = COCO2014Dataset(
        "/path_to_coco", "/path_to_preprocessed_folder", proc_total=8
    )
    print(coco[1])
1
