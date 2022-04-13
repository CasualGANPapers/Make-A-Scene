import torch
import numpy as np
import cv2
import face_alignment
import ssl
import matplotlib.pyplot as plt
ssl._create_default_https_context = ssl._create_unverified_context

BEARD = 0
BROW = 1
NOSE = 2
EYE = 3
MOUTH = 4


class FaceAlignmentPreprocessor:
    last_beard = 17
    last_brow = 27
    last_nose = 36
    last_eye = 48
    last_mouth = 68

    def __init__(self, n_classes=5, face_confidence=0.95, device="cuda"):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)
        self.face_confidence = face_confidence
        self.n_classes = n_classes
        self.class_idxs = {
            BEARD: torch.arange(0, self.last_beard),
            BROW: torch.arange(self.last_beard, self.last_brow),
            NOSE: torch.arange(self.last_brow, self.last_nose),
            EYE: torch.arange(self.last_nose, self.last_eye),
            MOUTH: torch.arange(self.last_eye, self.last_mouth),
        }

    def interpolate_face(self, face):
        interpolation = []
        for class_id in range(self.n_classes):
            part_interpolation = []
            part = face[self.class_idxs[class_id]]
            for idx, (i, j) in enumerate(zip(part, part[1:])):
                if self.class_idxs[class_id][idx] in (21, 41):  # to avoid that both eyes (or both brows) are connected
                    continue
                # print(self.class_idxs[class_id][idx])
                part_interpolation.extend(
                    list(np.round(np.linspace(i, j, 100)).astype(np.int32)) +
                    list(np.round(np.linspace(i, j, 100)).astype(np.int32) + [0, 1]) +
                    list(np.round(np.linspace(i, j, 100)).astype(np.int32) + [0, -1]) +
                    list(np.round(np.linspace(i, j, 100)).astype(np.int32) + [1, 0]) +
                    list(np.round(np.linspace(i, j, 100)).astype(np.int32) + [-1, 0])
                )
            interpolation.append(part_interpolation)
        return interpolation

    def process_image(self, img):
        img = img[:, :, ::-1]  # face_alignment work with BGR colorspace
        points = self.fa.get_landmarks(img)
        seg_mask = torch.zeros(*img.shape[:-1])
        if points is None:
            return seg_mask
        for face in points:
            face = face.astype(int)
            for class_id in range(self.n_classes):
                for point_id in self.class_idxs[class_id]:
                    try:
                        seg_mask[face[point_id, 1], face[point_id, 0]] = class_id + 1
                    except IndexError:
                        # Probably only part of the face on the image
                        pass
        return seg_mask  # F.one_hot(seg_mask.to(torch.long), num_classes=6)[..., 1:].permute(2, 0, 1)

    def process_image_interpolated(self, img):
        img = img[:, :, ::-1]  # face_alignment work with BGR colorspace
        faces = self.fa.face_detector.detect_from_image(img.copy())
        faces = list(filter(lambda face: face[-1] > self.face_confidence, faces))
        points = self.fa.get_landmarks(img, detected_faces=faces)
        seg_mask = torch.zeros(*img.shape[:-1])
        if points is None:
            return seg_mask
        for face in points:
            face = self.interpolate_face(face.astype(int))
            for class_id in range(self.n_classes):
                for point in face[class_id]:
                    try:
                        seg_mask[point[1], point[0]] = class_id + 1
                    except IndexError as e:
                        # Probably only part of the face on the image
                        pass
        boxes = [face[:-1] for face in faces]
        return seg_mask, boxes  # F.one_hot(seg_mask.to(torch.long), num_classes=6)[..., 1:].permute(2, 0, 1)

    def plot_face(self, seg_mask: torch.Tensor):
        plt.imshow(seg_mask.clamp(0, 1).detach().cpu().numpy(), cmap="gray")
        plt.show()

    def __call__(self, img):
        return self.process_image_interpolated(img)


if __name__ == "__main__":
    face_alignment_preprocessor = FaceAlignmentPreprocessor()
    img = cv2.imread("test.png")  # cv2 has other order of channels.
    print(img.shape)
    alignment, boxes = face_alignment_preprocessor(img)
    #face_alignment_preprocessor.plot_face(alignment)
    print(alignment.shape)
    print(boxes)
    torch.save(alignment, "alignment.pth")
