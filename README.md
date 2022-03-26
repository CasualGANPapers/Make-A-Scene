# Make-A-Scene - PyTorch
Pytorch implementation of Make-A-Scene: Scene-Based Text-to-Image Generation with Human Priors (https://arxiv.org/pdf/2203.13131.pdf)

<p align="center">
<img width="500" alt="results" src="https://user-images.githubusercontent.com/61938694/160241766-38daac29-6d07-4ff3-97ac-5b0f56e17271.png">
<em>Figure 1. from paper</em>
</p>

#### Note: this is work in progress. Everyone is happily invited to contribute

## Paper Description:
Make-A-Scene modifies the VQGAN framework. It makes heavy use of using semantic segmentation maps for extra conditioning. This enables more influence on the generation process. Morever, it also conditions on text. The main improvements are the following:
1. Segmentation condition: separate VQVAE is trained (VQ-SEG) + loss modified to a weighted binary cross entropy. (3.4)
2. VQGAN training (VQ-IMG) is extended by Face-Loss & Object-Loss (3.3 & 3.5)
3. Classifier Guidance for the autoregressive transformer (3.7)

## Training Pipeline
<p align="center">
<img width="500" alt="results" src="https://user-images.githubusercontent.com/61938694/160242667-fd82b900-b2df-4ffb-9cee-54660e502944.png">
<em>Figure 6. from paper</em>
</p>
