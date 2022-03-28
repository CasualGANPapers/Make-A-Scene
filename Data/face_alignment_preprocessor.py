import torch
import cv2
from skimage import io
import face_alignment

BEARD = 0
BROW = 1
NOSE = 2
EYE = 3
MOUTH = 4

class FaceAlignmentPreprocessor:
    last_beard = 17
    last_brow =  27
    last_nose = 36
    last_eye = 48
    last_mouth = 68
    def __init__(self,n_classes=5, device="cuda"):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)
        self.n_classes = n_classes
        self.class_idxs = {
                BEARD:  torch.arange(0, self.last_beard),
                BROW:  torch.arange(self.last_beard, self.last_brow),
                NOSE:  torch.arange(self.last_brow, self.last_nose),
                EYE:  torch.arange(self.last_nose, self.last_eye),
                MOUTH:  torch.arange(self.last_eye, self.last_mouth),
                }

    def process_image(self, img):
        points = self.fa.get_landmarks(img)
        seg_mask = torch.zeros(self.n_classes, *img.shape[:-1])
        if points is None:
            return seg_mask
        for face in points:
            face = face.astype(int)
            for class_id in range(self.n_classes):
                for point_id in self.class_idxs[class_id]:
                    try:
                        seg_mask[class_id, face[point_id, 1], face[point_id, 0]] = 1
                    except IndexError:
                        # Probably only part of the face on the image
                        pass
        return seg_mask

    def __call__(self, img):
        return self.process_image(img)

if __name__ == "__main__":
    face_alignment_preprocessor = FaceAlignmentPreprocessor()
    #img = cv2.imread("test.png")  cv2 has other order of channels.
    img = io.imread("test.png")
    print(img.shape)
    alignment = face_alignment_preprocessor(img)
    print(alignment.shape)
    torch.save(alignment, "alignment.pth")


