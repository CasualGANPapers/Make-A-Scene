import cv2
import numpy as np

THICKNESS = 1


def get_edges(masks):
    #!face contours are not used
    all_edges = np.zeros(masks.shape)
    for i, mask in enumerate(masks):
        edges = np.zeros(masks[0].shape)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE
        )
        edges = cv2.drawContours(edges, contours, -1, 1, THICKNESS)
        all_edges[i] = edges
    return all_edges
