import torch


def collate_fn(batch, need_seg=False):
    images = torch.stack([i[0] for i in batch], dim=0)
    if need_seg:
        segmentation_maps = torch.stack([torch.Tensor(i[1]).permute(2, 0, 1) for i in batch], dim=0)
    else:
        segmentation_maps = []
    object_boxes = [list(map(lambda bbox: list(map(int, bbox))[:-1], i[2])) for i in batch]
    face_boxes = [list(map(lambda bbox: list(map(int, bbox))[:-1], i[3])) for i in batch]
    captions = [i[4] for i in batch]
    return [images, segmentation_maps, object_boxes, face_boxes, captions]


def collate_fn_(batch):
    images, segmentation_maps = None, None
    object_boxes, face_boxes, captions = [], [], []
    for i in batch:
        images = i[0] if images is None else torch.stack([images, i[0]], dim=0)
        segmentation_maps = torch.Tensor(i[1]).permute(2, 0, 1) if segmentation_maps is None else torch.stack([segmentation_maps, torch.Tensor(i[1]).permute(2, 0, 1)], dim=0)
        object_boxes.append(i[2][:-1])
        face_boxes.append(i[3][:-1])
        captions.append(i[4])
    return [images, segmentation_maps, object_boxes, face_boxes, captions]

def change_requires_grad(model, state):
    for parameter in model.parameters():
        parameter.requires_grad = state

