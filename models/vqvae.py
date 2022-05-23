import torch
import torch.nn.functional as F
from torch import nn
from .modules import Encoder, Decoder
from .modules import Codebook


class VQBASE(nn.Module):
    def __init__(self, ddconfig, n_embed, embed_dim, init_steps):
        super(VQBASE, self).__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = Codebook(n_embed, embed_dim, beta=0.25, init_steps=init_steps)  # TODO: change length_one_epoch
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

