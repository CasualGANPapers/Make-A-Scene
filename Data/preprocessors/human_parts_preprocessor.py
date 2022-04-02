import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

HUMAN_PARSER_DIR = "../../../Self-Correction-Human-Parsing"
sys.path.append(HUMAN_PARSER_DIR)
import networks
from simple_extractor import get_palette, dataset_settings
from collections import OrderedDict
from utils.transforms import transform_logits, get_affine_transform


class HumanPartsPreprocessor:
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
                transforms.ToTensor(),
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

    def segment_image(self, image):
        h, w, _ = image.shape

        # Get person center and scale
        person_center, scale = self._box2cs([0, 0, w - 1, h - 1])
        c = person_center
        s = scale
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        image = cv2.warpAffine(
            image,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
        upsample_output = self.upsample(output[0][-1][0].unsqueeze(0))
        upsample_output = upsample_output.squeeze()
        upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

        logits_result = transform_logits(
            upsample_output.data.cpu().numpy(), c, s, w, h, input_size=self.input_size
        )
        parsing_result = np.argmax(logits_result, axis=2)
        encoded = torch.from_numpy(parsing_result)
        return encoded

    def __call__(self, image):
        return self.segment_image(image)


if __name__ == "__main__":
    human_processor = HumanPartsPreprocessor()
    img = cv2.imread("two.png")
    masks = human_processor(img)
    print(masks)
    print(masks.shape)
