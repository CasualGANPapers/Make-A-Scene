
def check_bbox(bbox):
    """Check if bbox minimums are lesser then maximums"""
    x_min, y_min, x_max, y_max = bbox[:4]
    bbox[0] = 0 if x_min < 0 else x_min
    bbox[1] = 0 if y_min < 0 else y_min
    bbox[2] = 511 if x_max >511 else x_max
    bbox[3] = 511 if y_max >511 else y_max
    if x_max <= x_min:
        return None
    if y_max <= y_min:
        return None
    if y_max - y_min < 16 or x_max - x_min < 16:
        # print(f"removing bbox: {bbox}")
        return None

    return bbox


def check_bboxes(bboxes):
    """Check if bboxes boundaries are in range 0, 1 and minimums are lesser then maximums"""
    _bboxes = []
    for bbox in bboxes:
        check = check_bbox(bbox)
        if check is not None:
            _bboxes.append(bbox)
        # else:
        #     print(f"Removing bbox: {bbox}")
    return _bboxes
