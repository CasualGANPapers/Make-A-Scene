import argparse
import logging
import os
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.utils as vutils
from tqdm import tqdm
from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer


class BCELossWithQuant(nn.Module):
    def __init__(self, codebook_weight=1.):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.weight = torch.ones(args.image_channels).index_fill(0, torch.arange(153, 158), 20)

    def forward(self, qloss, target, prediction):
        bce_loss = F.binary_cross_entropy_with_logits(prediction, target, weight=self.weight)
        loss = bce_loss + self.codebook_weight * qloss
        return loss


class VQSegmentationModel(nn.Module):
    def __init__(self, ddconfig, args):
        super().__init__()
        self.register_buffer("colorize", torch.randn(3, args.image_channels, 1, 1))
        self.loss = BCELossWithQuant()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(args.num_codebook_vectors, args.latent_dim, beta=0.25)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1)

    def setup(self, args):
        path = os.path.join("checkpoints", self.run_name)
        if os.path.exists(path) and args.start_from_epoch == 0:
            raise Exception(f"{self.run_name} is already a directory, please choose a different name for your run!")
        else:
            if args.start_from_epoch == 0:
                os.mkdir(path)
                os.mkdir(os.path.join(path, "results"))
            else:
                self.load_checkpoint(args.start_from_epoch)
            logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, force=True,
                                datefmt="%I:%M:%S",
                                handlers=[logging.FileHandler(os.path.join(path, "log.txt")), logging.StreamHandler()])

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def load_checkpoint(self, epoch):
        path = os.path.join("checkpoints", self.run_name, f"epoch_{epoch}")
        assert isinstance(epoch, int)
        self.encoder.load_state_dict(torch.load(os.path.join(path, "encoder.pt")))
        self.decoder.load_state_dict(torch.load(os.path.join(path, "decoder.pt")))
        self.codebook.load_state_dict(torch.load(os.path.join(path, "codebook.pt")))
        self.quant_conv.load_state_dict(torch.load(os.path.join(path, "quant_conv.pt")))
        self.post_quant_conv.load_state_dict(torch.load(os.path.join(path, "post_quant_conv.pt")))
        logging.info(f"Loaded all models from epoch {epoch} (without Discriminator!")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def training(self, args):
        # load_data() still needs to be defined
        dataset = load_data(args)
        for epoch in range(args.start_from_epoch, args.epochs):
            with tqdm(range(len(dataset))) as pbar:
                for i, imgs in zip(pbar, dataset):
                    imgs = imgs.to(device=args.device)
                    imgs_rec, qloss = self(imgs)
                    loss_vq = self.loss(qloss, imgs, imgs_rec)

                    self.opt_vq.zero_grad()
                    loss_vq.backward(retain_graph=True)
                    self.opt_vq.step()

                    if i % 10 == 0:
                        with torch.no_grad():
                            both = torch.cat((imgs[:4], imgs_rec.add(1).mul(0.5)[:4]))
                            vutils.save_image(both, "results" + f'/{epoch}_{i}.jpg', nrow=4)

                if args.save_models:
                    path = os.path.join("checkpoints", self.run_name)
                    os.makedirs(path, exist_ok=True)
                    epoch_path = os.path.join(path, f"epoch_{epoch}")
                    os.makedirs(os.path.join(epoch_path), exist_ok=True)
                    torch.save(self.encoder.state_dict(), os.path.join(epoch_path, "encoder.pt"))
                    torch.save(self.decoder.state_dict(), os.path.join(epoch_path, "decoder.pt"))
                    torch.save(self.codebook.state_dict(), os.path.join(epoch_path, "codebook.pt"))
                    torch.save(self.quant_conv.state_dict(), os.path.join(epoch_path, "quant_conv.pt"))
                    torch.save(self.post_quant_conv.state_dict(), os.path.join(epoch_path, "post_quant_conv.pt"))

    @torch.no_grad()
    def log_images(self, batch):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQ-SEG")
    parser.add_argument('--run-name', type=str, default='default',
                        help='Give this training run a name (default: current timestamp)')
    parser.add_argument('--latent-dim', type=int, default=512, help='latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024,
                        help='number of codebook vectors (default: 1024)')
    parser.add_argument('--beta', type=float, default=0.25, help='commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=159, help='number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', metavar='Path',
                        help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", metavar='cuda', help='which device the training is on')
    parser.add_argument('--batch-size', type=int, default=5, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=5e-5, metavar='LR', help='learning rate')
    parser.add_argument('--start-from-epoch', type=int, default=0,
                        help='Use a saved checkpoint, 0 means no checkpoint (default: 0)')
    parser.add_argument('--save-models', type=bool, default=True, help='Save all models each epoch (default: True)')

    args = parser.parse_args()

    # taken from configs/sflickr_cond_stage.yaml
    dd_config = {"double_z": False,
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

    vq_seg = VQSegmentationModel(dd_config, args)
