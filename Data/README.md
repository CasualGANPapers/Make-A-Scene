# Data Aggregation

## Segmentation Dataset

"VQ-SEG and VQ-IMG are trained on CC12m, CC, and MS-COCO." -> For the segmentation
process we first need to convert all the 3 datasets to segmentation datasets using
the 3 models described below:

- Panoptic: https://github.com/facebookresearch/detectron2
- Human Parts: https://github.com/PeikeLi/Self-Correction-Human-Parsing
- Human Face: https://github.com/1adrianb/face-alignment

VQ-SEG was trained to have 158 categories
- 133 panoptic
- 20 human parts
- 5 human face (eye-brows, eyes, nose, outer-mouth, inner-mouth)
	
These 3 models will be used to construct the dataset for the segmentation maps.

The data aggregation pipeline should do the following:
1. Take in an image (optionally entire dataloader)
2. seg_panoptic = detectron2(x) # use only 133 panoptic
3. seg_human = human_parsing(x) # the one trained on LIP
4. seg_face = human_face(x)
5. Concatenate along channel axis
6. Add one channel for edges between objects
7. return segmentation map

