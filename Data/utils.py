
def check_bbox(bbox):
    """Check if bbox minimums are lesser then maximums"""
    x_min, y_min, x_max, y_max = bbox[:4]
    if x_max <= x_min:
        return None
    if y_max <= y_min:
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