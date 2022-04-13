import numpy as np
import os
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.multiprocessing as mp
import warnings
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

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
        force_preprocessing=True,
        proc_total=1,
    ):
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

        img_names_path = os.path.join(preprocessed_folder, "img_names.npz")
        if os.path.exists(img_names_path) and not force_preprocessing:
            self.img_names = np.load(img_names_path)["img_names"]
        else:
            self.parse_image_names(image_dirs)
            self.img_names = np.load(img_names_path)["img_names"]

        if not os.path.exists(preprocessed_folder) or force_preprocessing:
            self.preprocess_dataset()

    def parse_image_names(self, img_dirs):
        img_names = []
        for directory in img_dirs:
            for filename in os.listdir(os.path.join(self.root, directory)):
                if os.path.splitext(filename)[1] in [".jpg", ".png"]:
                    img_names.append(os.path.join(directory, filename))
        np.savez(
            os.path.join(self.preprocessed_folder, "img_names"), img_names=img_names
        )

    def preprocess_dataset(self):
        assert torch.cuda.is_available(), "GPU required for preprocessing"
        if self.proc_total > torch.cuda.device_count():
            warnings.warn(
                "Do not recommended to set more processes, than you have GPUs"
            )
        if self.proc_total == 1:
            self.detectron2 = Detectron2()
            self.human_parsing = HumanParts()
            self.human_face = HumanFace()
            for img_name in tqdm(self.img_names):
                self.preprocess_single_image(img_name)
        else:
            self.detectron2 = [None] * self.proc_total
            self.human_parsing = [None] * self.proc_total
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
                names = np.load(f"tmp_names_{proc_id}.npz")["names"]
                os.remove(f"tmp_names_{proc_id}.npz")
                self.img_names.extend(names)
            print(len(self.img_names))
            np.savez(
                os.path.join(self.preprocessed_folder, "img_names"), img_names=self.img_names
            )

    def preprocess_single_process(self, proc_id,):
        correct_names = []
        device = f"cuda:{proc_id%torch.cuda.device_count()}"
        torch.cuda.set_device(  # https://github.com/pytorch/pytorch/issues/21819#issuecomment-553310128
            proc_id % self.proc_total
        )
        self.detectron2[proc_id] = Detectron2(device=device)
        self.human_parsing[proc_id] = HumanParts(device=device)
        self.human_face[proc_id] = HumanFace(device=device)
        img_names = self.img_names[proc_id :: self.proc_total]
        bar = tqdm(img_names) if proc_id == 0 else img_names
        for img_name in bar:
            if self.preprocess_single_image(img_name, proc_id,):
                correct_names.append(img_name)
        np.savez(f"tmp_names_{proc_id}", names=correct_names)

    def preprocess_single_image(self, img_name, proc_id=None, correct_img_names=None):
        detectron2 = self.detectron2 if proc_id is None else self.detectron2[proc_id]
        human_parsing = (
            self.human_parsing if proc_id is None else self.human_parsing[proc_id]
        )
        human_face = self.human_face if proc_id is None else self.human_face[proc_id]

        image = cv2.imread(os.path.join(self.root, img_name))
        #if image.shape[0]<256 or image.shape[1]<256:
        #    return False

        seg_panoptic = detectron2(image).to(torch.int32).cpu().numpy()
        seg_human = human_parsing(image).to(torch.int32).cpu().numpy()
        seg_face = human_face(image)
        seg_edges = get_edges(seg_panoptic, seg_human, seg_face)

        seg_panoptic = np.array(seg_panoptic // 1000, dtype=np.uint8)
        seg_human = np.array(seg_human, dtype=np.uint8)
        seg_face = np.array(seg_face, dtype=np.uint8)
        seg_edges = np.array(seg_edges, dtype=np.bool)

        save_path = os.path.join(
            self.preprocessed_folder, "segmentations", os.path.splitext(img_name)[0]
        )
        try:
            np.savez_compressed(
                save_path,
                seg_panoptic=seg_panoptic,
                seg_human=seg_human,
                seg_face=seg_face,
                seg_edges=seg_edges,
            )
        except FileNotFoundError:
            os.makedirs(os.path.dirname(save_path))
            np.savez_compressed(
                save_path,
                seg_panoptic=seg_panoptic,
                seg_human=seg_human,
                seg_face=seg_face,
                seg_edges=seg_edges,
            )
        return True

    def load_segmentation(self, img_name):
        data = np.load(
            os.path.join(
                self.preprocessed_folder,
                "segmentations",
                os.path.splitext(img_name)[0] + ".npz",
            )
        )
        seg_panoptic = F.one_hot(
            torch.from_numpy(data["seg_panoptic"] + 1).to(torch.long), num_classes=134
        )[..., 1:]
        seg_human = F.one_hot(
            torch.from_numpy(data["seg_human"] + 1).to(torch.long), num_classes=21
        )[..., 1:]
        seg_face = F.one_hot(
            torch.from_numpy(data["seg_face"]).to(torch.long), num_classes=6
        )[..., 1:]
        seg_edges = torch.from_numpy(data["seg_edges"]).unsqueeze(-1)

        seg_map = torch.cat(
            [seg_panoptic, seg_human, seg_face, seg_edges], dim=-1
        )#.permute(2, 0, 1)
        return seg_map

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        try:
            segmentation = self.load_segmentation(img_name)
        except FileNotFoundError:
            self.preprocess_single_image(img_name)
            segmentation = self.load_segmentation(img_name)
        image = self.get_image(img_name)
        data = self.transforms(image=image, mask=segmentation.numpy())
        return data["image"], data["mask"].float()

    def get_image(self, img_name):
        raise NotImplementedError

    def __len__(self):
        return len(self.img_names)


class COCO2014Dataset(PreprocessedDataset):
    def __init__(self, root, preprocessed_folder, **kwargs):
        super().__init__(
            root=root,
            image_dirs=["images/train2014"],
            preprocessed_folder=preprocessed_folder,
            **kwargs,
        )

    def get_image(self, img_name):
        path = os.path.join(self.root, img_name)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


if __name__ == "__main__":
    coco = COCO2014Dataset(
        "/path_to_coco", "/path_to_preprocessed_folder", proc_total=8
    )
    print(coco[1])
1
