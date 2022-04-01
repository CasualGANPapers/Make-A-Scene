import argparse
import torch
import torch.nn.functional as F
from torch import nn
from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer
import pytorch_lightning as pl
from dataset_preprocessor import COCO2014Dataset
from torch.utils.data import DataLoader
import torchvision.utils as vutils


pl.seed_everything(69696969)


class BCELossWithQuant(nn.Module):
    def __init__(self, codebook_weight=1.):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.register_buffer("weight", torch.ones(args.image_channels).index_fill(0, torch.arange(153, 158), 20))

    def forward(self, qloss, target, prediction):
        bce_loss = F.binary_cross_entropy_with_logits(prediction.permute(0, 2, 3, 1), target.permute(0, 2, 3, 1),
                                                      pos_weight=self.weight)
        loss = bce_loss + self.codebook_weight * qloss
        return loss


class VQBASE(pl.LightningModule):
    def __init__(self, args):
        super(VQBASE, self).__init__()
        self.encoder = Encoder(**args.dd_config).to(args.device)
        self.decoder = Decoder(**args.dd_config).to(args.device)
        self.quantize = VectorQuantizer(args.num_codebook_vectors, args.latent_dim, beta=0.25).to(args.device)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(args.device)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(args.device)
        self.learning_rate = args.learning_rate

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss = self.quantize(h)
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


class VQSEG(VQBASE):
    def __init__(self, args):
        super().__init__(args)
        self.register_buffer("colorize", torch.randn(3, args.image_channels, 1, 1))
        self.loss = BCELossWithQuant()

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        _, seg = batch
        seg_rec, qloss = self(seg)
        loss_vq = self.loss(qloss, seg, seg_rec)
        self.log("vq_loss", loss_vq, prog_bar=True)

        if batch_idx % 10 == 0:
            with torch.no_grad():
                seg_rgb, seg_rec_rgb = self.log_images(seg, seg_rec)
                both = torch.cat((self.to_rgb(seg), seg_rgb, self.to_rgb(seg_rec), seg_rec_rgb))
                grid = vutils.make_grid(both, nrow=4)
                vutils.save_image(both, f"./results/{self.current_epoch}_{batch_idx}.jpg", nrow=4)
                self.logger.experiment.add_image('Generated Images', grid, f"{self.current_epoch}_{batch_idx}")

        return loss_vq

    def validation_step(self, batch, batch_idx):
        _, seg = batch
        seg_rec, qloss = self(seg)
        loss_vq = self.loss(qloss, seg, seg_rec)
        self.log("vq_loss", loss_vq, prog_bar=True)
        return loss_vq

    def to_rgb(self, x):
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

    @torch.no_grad()
    def log_images(self, seg, seg_rec):
        seg_rec = torch.argmax(seg_rec, dim=1, keepdim=True)
        seg_rec = F.one_hot(seg_rec, num_classes=seg.shape[1])
        seg_rec = seg_rec.squeeze(1).permute(0, 3, 1, 2).float()
        return self.to_rgb(seg), self.to_rgb(seg_rec)


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
