import cv2
import numpy as np

THICKNESS = 1


def get_edges(masks):
    #!face contours are not used
    edges = np.zeros(masks.shape)
    contours, _ = cv2.findContours(
        masks, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE
    )
    edges = cv2.drawContours(edges, contours, -1, 1, THICKNESS)
    return edges
