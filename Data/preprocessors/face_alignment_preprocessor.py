import torch
import numpy as np
import cv2
import face_alignment
import ssl
import matplotlib.pyplot as plt
ssl._create_default_https_context = ssl._create_unverified_context
from time import time

BEARD = 0
BROW = 1
NOSE = 2
EYE = 3
MOUTH = 4


class FaceAlignmentPreprocessor:
    proc_type = "face"
    last_beard = 17
    last_brow = 27
    last_nose = 36
    last_eye = 48
    last_mouth = 68

    def __init__(self, n_classes=5, face_confidence=0.95, device="cuda"):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector_kwargs={"filter_threshold":face_confidence}, device=device)
        self.face_confidence = face_confidence
        self.n_classes = n_classes
        self.class_idxs = {
            BEARD: (0, self.last_beard),
            BROW: (self.last_beard, self.last_brow),
            NOSE: (self.last_brow, self.last_nose),
            EYE: (self.last_nose, self.last_eye),
            MOUTH: (self.last_eye, self.last_mouth),
        }
        #self.class_idxs = {
        #    BEARD: torch.arange(0, self.last_beard),
        #    BROW: torch.arange(self.last_beard, self.last_brow),
        #    NOSE: torch.arange(self.last_brow, self.last_nose),
        #    EYE: torch.arange(self.last_nose, self.last_eye),
        #    MOUTH: torch.arange(self.last_eye, self.last_mouth),
        #}


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
    def process_image_interpolated(self, imgs: np.array):
        # imgs should be numpy b x c x h x w
        imgs = imgs.flip([1])  # face_alignment works with BGR colorspace
        faces = self.fa.face_detector.detect_from_batch(imgs)
        # faces = list(filter(lambda face: face[-1] > self.face_confidence, faces))
        faces = list(map(lambda img: list(filter(lambda face: face[-1] > self.face_confidence, img)), faces))
        batched_points = self.fa.get_landmarks_from_batch(imgs, detected_faces=faces)
        seg_mask = np.zeros((imgs.shape[0], *imgs.shape[2:]))
        if batched_points is None:
            return seg_mask
        for i, points in enumerate(batched_points):
            for face in points:
                face = self.interpolate_face(face.astype(int))
                for class_id in range(self.n_classes):
                    for point in face[class_id]:
                        try:
                            seg_mask[i, point[1], point[0]] = class_id + 1
                        except IndexError as e:
                            # Probably only part of the face on the image
                            pass
        boxes = [[face[:-1] for face in faces_in_image] for faces_in_image in faces]
        return seg_mask, boxes  # F.one_hot(seg_mask.to(torch.long), num_classes=6)[..., 1:].permute(2, 0, 1)

    def draw_interpolated_face(self,mask, face):
        for class_id in range(self.n_classes):
            start, stop = self.class_idxs[class_id]
            if class_id not in (EYE, BROW):
                cv2.drawContours(mask, [face[start: stop]], 0 , (class_id+1), 1)
            else:
                step = (stop-start)//2
                cv2.drawContours(mask, [face[start:start+step]], 0, (class_id+1), 1)
                cv2.drawContours(mask, [face[start+step: stop]], 0, (class_id+1), 1)
        return mask

    def process_image_interpolated_fast(self, imgs: np.array):
        # imgs should be numpy b x c x h x w
        t = time()
        times = []
        imgs = imgs.flip([1])  # face_alignment works with BGR colorspace
        faces = self.fa.face_detector.detect_from_batch(imgs)
        times.append([time()-t])
        t = time()
        batched_points = self.fa.get_landmarks_from_batch(imgs, detected_faces=faces)
        seg_mask = np.zeros((imgs.shape[0], *imgs.shape[2:]))
        times.append([time()-t])
        t = time()
        for image_idx, points in enumerate(batched_points):
            for face in points:
                seg_mask[image_idx] = self.draw_interpolated_face(seg_mask[image_idx], face.astype(int))
        boxes = [[face[:-1] for face in faces_in_image] for faces_in_image in faces]
        times.append([time()-t])
        t = time()
        #print("HERE!!!", times)
        return seg_mask, boxes  # F.one_hot(seg_mask.to(torch.long), num_classes=6)[..., 1:].permute(2, 0, 1)

    def process_image_interpolated_old(self, img):
        img = img[:, :, ::-1]  # face_alignment work with BGR colorspace
        faces = self.fa.face_detector.detect_from_image(img.copy())
        faces = list(filter(lambda face: face[-1] > self.face_confidence, faces))
        points = self.fa.get_landmarks(img, detected_faces=faces)
        seg_mask = np.zeros(img.shape[:-1])
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
        data = {}
        mask, boxes = self.process_image_interpolated_fast(img)
        # mask, boxes = self.process_image_interpolated_old(img)
        data["seg_face"] = mask.astype(np.uint8)
        data["box_face"] = boxes
        return data


if __name__ == "__main__":
    face_alignment_preprocessor = FaceAlignmentPreprocessor()
    img = cv2.imread("humans.jpg")  # cv2 has other order of channels.
    print(img.shape)
    img = np.repeat(img.transpose(2, 0, 1)[None, ...], 5, axis=0)
    data = face_alignment_preprocessor(img)
    #face_alignment_preprocessor.plot_face(alignment)
    print(data["seg_face"].shape)
    print(len(data["box_face"]))
    print(data["box_face"])
    # torch.save(alignment, "alignment.pth")
