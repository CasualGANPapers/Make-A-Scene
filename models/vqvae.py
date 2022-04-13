import argparse
import torch
import torch.nn.functional as F
from torch import nn
from .modules import Encoder, Decoder
#from taming.modules.vqvae.quantize import VectorQuantizer
from .modules import VectorQuantizer2 as VectorQuantizer
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.utils as vutils


class VQBASE(nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super(VQBASE, self).__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff = self.encode(input)
        dec = self.decode(quant)
        return dec, diff



def train(args):
    vqseg = VQSEG(args)
    dataset = COCO2014Dataset("./tmpdb/", "./tmpdb/preprocessed_folder")
    data_loader = DataLoader(dataset, batch_size=args.batch_size)
    trainer = pl.Trainer(accelerator="gpu", devices=1, check_val_every_n_epoch=100, max_epochs=50,
                         num_sanity_val_steps=0)
    trainer.fit(vqseg, data_loader, data_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQ-SEG")
    parser.add_argument('--run-name', type=str, default='default',
                        help='Give this training run a name (default: current timestamp)')
    parser.add_argument('--latent-dim', type=int, default=32, help='latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024,
                        help='number of codebook vectors (default: 1024)')
    parser.add_argument('--beta', type=float, default=0.25, help='commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=159, help='number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', metavar='Path',
                        help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", metavar='cuda', help='which device the training is on')
    parser.add_argument('--batch-size', type=int, default=2, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=1e-5, metavar='LR', help='learning rate')
    parser.add_argument('--start-from-epoch', type=int, default=0,
                        help='Use a saved checkpoint, 0 means no checkpoint (default: 0)')
    parser.add_argument('--save-models', type=bool, default=True, help='Save all models each epoch (default: True)')

    args = parser.parse_args()

    args.dd_config = {"double_z": False,
                      "z_channels": 256,
                      "resolution": 256,
                      "in_channels": 160,
                      "out_ch": 160,
                      "ch": 128,
                      "ch_mult": [1, 1, 2, 2, 4],
                      "num_res_blocks": 2,
                      "attn_resolutions": 16,
                      "dropout": 0.0
                      }

    train(args)
