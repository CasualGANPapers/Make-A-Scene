# Make-A-Scene - PyTorch
Pytorch implementation (unofficial) of Make-A-Scene: Scene-Based Text-to-Image Generation with Human Priors

<p align="center">
<img width="500" alt="results" src="https://user-images.githubusercontent.com/61938694/160241766-38daac29-6d07-4ff3-97ac-5b0f56e17271.png">
<em>Figure 1. from paper</em>
</p>

## Note: this is work in progress. 
We are at training stage! The process can be followed in the Discord-Channel on the LAION Discord https://discord.gg/DghvZDKu.
The data preprocessing has been finished as well as training VQSEG. We are currently training VQIMG. Training checkpoints will be released soon with demos.
The transformer implementation is in progess and will hopefully be started to train as soon as VQIMG finishes.

## Demo
VQIMG: https://colab.research.google.com/drive/1SPyQ-epTsAOAu8BEohUokN4-b5RM_TnE?usp=sharing

## Paper Description
Make-A-Scene modifies the VQGAN framework. It makes heavy use of using semantic segmentation maps for extra conditioning. This enables more influence on the generation process. Morever, it also conditions on text. The main improvements are the following:
1. Segmentation condition: separate VQVAE is trained (VQ-SEG) + loss modified to a weighted binary cross entropy. (3.4)
2. VQGAN training (VQ-IMG) is extended by Face-Loss & Object-Loss (3.3 & 3.5)
3. Classifier Guidance for the autoregressive transformer (3.7)

## Training Pipeline
<p align="center">
<img width="500" alt="results" src="https://user-images.githubusercontent.com/61938694/160242667-fd82b900-b2df-4ffb-9cee-54660e502944.png">
<em>Figure 6. from paper</em>
</p>

## What needs to be done?
Refer to the different folders to see details.
- [X] [VQ-SEG](https://github.com/CasualGANPapers/Make-A-Scene/tree/main/VQ-SEG)
- [ ] [VQ-IMG](https://github.com/CasualGANPapers/Make-A-Scene/tree/main/VQ-IMG)
- [ ] [Transformer]()
- [X] [Data Aggregation](https://github.com/CasualGANPapers/Make-A-Scene/tree/main/Data)

## Citation
```bibtex
@misc{https://doi.org/10.48550/arxiv.2203.13131,
  doi = {10.48550/ARXIV.2203.13131},
  url = {https://arxiv.org/abs/2203.13131},
  author = {Gafni, Oran and Polyak, Adam and Ashual, Oron and Sheynin, Shelly and Parikh, Devi and Taigman, Yaniv},
  title = {Make-A-Scene: Scene-Based Text-to-Image Generation with Human Priors},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
