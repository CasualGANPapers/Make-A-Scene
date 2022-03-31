import cv2
import numpy as np

THICKNESS = 1


def get_edges(panoptic, human, face):
    #!face contours are not used
    edges = np.zeros(panoptic.shape)
    panoptic_contours, _ = cv2.findContours(
        panoptic, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE
    )
    human_contours, _ = cv2.findContours(
        human, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE
    )

    edges = cv2.drawContours(edges, panoptic_contours, -1, 1, THICKNESS)
    edges = cv2.drawContours(edges, human_contours, -1, 1, THICKNESS)
    return edges
