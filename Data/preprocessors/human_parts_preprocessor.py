import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchgeometry import warp_affine

HUMAN_PARSER_DIR = "/home/ubuntu/MakeAScene/Self-Correction-Human-Parsing"
sys.path.append(HUMAN_PARSER_DIR)
import networks
from simple_extractor import get_palette, dataset_settings
from collections import OrderedDict
from utils.transforms import get_affine_transform
from .edge_extractor import get_edges
import torchvision


def transform_logits(logits, center, scale, width, height, input_size):
    trans = torch.Tensor(get_affine_transform(center, scale, 0, input_size, inv=1)).expand(logits.shape[0], -1, -1)
    target_logits = warp_affine(logits, trans, (int(width), int(height)))
    return target_logits


class HumanPartsPreprocessor:
    proc_type = "human"
    def __init__(
        self,
        weights=HUMAN_PARSER_DIR + "/checkpoints/final.pth",
        device="cuda",
    ):
        self.device = device

        dataset_info = dataset_settings["lip"]
        self.num_classes = dataset_info["num_classes"]
        self.input_size = dataset_info["input_size"]
        self.model = networks.init_model(
            "resnet101", num_classes=self.num_classes, pretrained=None
        )
        self.aspect_ratio = self.input_size[1] * 1.0 / self.input_size[0]

        state_dict = torch.load(weights)["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        self.model.to(device)

        self.transform = transforms.Compose(
            [
                # transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]
                ),
            ]
        )
        self.upsample = torch.nn.Upsample(
            size=self.input_size, mode="bilinear", align_corners=True
        )

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def segment_image(self, imgs: np.array):
        # imgs should be numpy b x c x h x w
        #imgs = torch.Tensor(imgs)
        b, _, h, w = imgs.shape  # we can assume b x 3 x 512 x 512
                                 # check if h, w in correct order
        # Get person center and scale
        #person_center, scale = self._box2cs([0, 0, w - 1, h - 1])
        #c = person_center
        #s = scale
        #r = 0
        #trans = torch.Tensor(get_affine_transform(person_center, s, r, self.input_size)).expand(b, -1, -1)
        #imgs = warp_affine(imgs, trans, (int(self.input_size[1]), int(self.input_size[0])))
        imgs = torchvision.transforms.functional.resize(imgs, [self.input_size[1], self.input_size[0]])

        imgs = self.transform(imgs/255.).to(self.device)

        with torch.no_grad():
            output = self.model(imgs)
        upsample_output = self.upsample(output[0][-1])  # reshapes from 1, 20, 119, 119 to 1, 20, 473, 473

        #logits_result = transform_logits(upsample_output.cpu(), c, s, w, h, input_size=self.input_size)
        logits_result = torchvision.transforms.functional.resize(upsample_output, [h, w])
        mask = logits_result.argmax(dim=1)
        return mask.cpu().numpy()

    def __call__(self, imgs):
        data = {}
        mask = self.segment_image(imgs)
        edges = get_edges(mask)
        data["seg_human"] = mask.astype(np.uint8)
        data["edges"] = edges.astype(bool)
        return data


if __name__ == "__main__":
    human_processor = HumanPartsPreprocessor()
    img = cv2.imread("humans.jpg")
    img = torch.randint_like(torch.Tensor(img), 0, 255, dtype=torch.float).permute(2, 0, 1).expand(5, -1, -1, -1).numpy()
    masks = human_processor(img)
    print(masks)
