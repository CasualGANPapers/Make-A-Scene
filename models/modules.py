# Taken from https://github.com/CompVis/taming-transformers
# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fast_pytorch_kmeans import KMeans
from torch import einsum
import torch.distributed as dist
from einops import rearrange


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Encoder(nn.Module):
    """
    Encoder of VQ-GAN to map input batch of images to latent space.
    Dimension Transformations:
    3x256x256 --Conv2d--> 32x256x256
    for loop:
              --ResBlock--> 64x256x256 --DownBlock--> 64x128x128
              --ResBlock--> 128x128x128 --DownBlock--> 128x64x64
              --ResBlock--> 256x64x64  --DownBlock--> 256x32x32
              --ResBlock--> 512x32x32
    --ResBlock--> 512x32x32
    --NonLocalBlock--> 512x32x32
    --ResBlock--> 512x32x32
    --GroupNorm-->
    --Swish-->
    --Conv2d-> 256x32x32
    """

    def __init__(self, in_channels=3, channels=[128, 128, 128, 256, 512, 512], attn_resolutions=[32], resolution=512, dropout=0.0, num_res_blocks=2, z_channels=256, **kwargs):
        super(Encoder, self).__init__()
        layers = [nn.Conv2d(in_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResnetBlock(in_channels=in_channels, out_channels=out_channels, dropout=0.0))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(AttnBlock(in_channels))
            if i < len(channels) - 2:
                layers.append(Downsample(channels[i + 1], with_conv=True))
                resolution //= 2
        layers.append(ResnetBlock(in_channels=channels[-1], out_channels=channels[-1], dropout=0.0))
        layers.append(AttnBlock(channels[-1]))
        layers.append(ResnetBlock(in_channels=channels[-1], out_channels=channels[-1], dropout=0.0))
        layers.append(Normalize(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], z_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# class Encoder(nn.Module):
#     def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
#                  attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
#                  resolution, z_channels, double_z=True, **ignore_kwargs):
#         super().__init__()
#         self.ch = ch
#         self.temb_ch = 0
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         self.resolution = resolution
#         self.in_channels = in_channels
#
#         # downsampling
#         self.conv_in = torch.nn.Conv2d(in_channels,
#                                        self.ch,
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1)
#
#         curr_res = resolution
#         in_ch_mult = (1,)+tuple(ch_mult)
#         self.down = nn.ModuleList()
#         for i_level in range(self.num_resolutions):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_in = ch*in_ch_mult[i_level]
#             block_out = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks):
#                 block.append(ResnetBlock(in_channels=block_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 if curr_res in attn_resolutions:
#                     attn.append(AttnBlock(block_in))
#             down = nn.Module()
#             down.block = block
#             down.attn = attn
#             if i_level != self.num_resolutions-1:
#                 down.downsample = Downsample(block_in, resamp_with_conv)
#                 curr_res = curr_res // 2
#             self.down.append(down)
#
#         # middle
#         self.mid = nn.Module()
#         self.mid.block_1 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#         self.mid.attn_1 = AttnBlock(block_in)
#         self.mid.block_2 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#
#         # end
#         self.norm_out = Normalize(block_in)
#         self.conv_out = torch.nn.Conv2d(block_in,
#                                         2*z_channels if double_z else z_channels,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)
#
#
#     def forward(self, x):
#         #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)
#
#         # timestep embedding
#         temb = None
#
#         # downsampling
#         hs = [self.conv_in(x)]
#         for i_level in range(self.num_resolutions):
#             for i_block in range(self.num_res_blocks):
#                 h = self.down[i_level].block[i_block](hs[-1], temb)
#                 if len(self.down[i_level].attn) > 0:
#                     h = self.down[i_level].attn[i_block](h)
#                 hs.append(h)
#             if i_level != self.num_resolutions-1:
#                 hs.append(self.down[i_level].downsample(hs[-1]))
#
#         # middle
#         h = hs[-1]
#         h = self.mid.block_1(h, temb)
#         h = self.mid.attn_1(h)
#         h = self.mid.block_2(h, temb)
#
#         # end
#         h = self.norm_out(h)
#         h = nonlinearity(h)
#         h = self.conv_out(h)
#         return h


class Decoder(nn.Module):
    def __init__(self, out_channels=3, channels=[128, 128, 128, 256, 512, 512], attn_resolutions=[32], resolution=512, dropout=0.0, num_res_blocks=2, z_channels=256, **kwargs):
        super(Decoder, self).__init__()
        ch_mult = channels[1:]
        num_resolutions = len(ch_mult)
        block_in = ch_mult[num_resolutions - 1]
        curr_res = resolution// 2 ** (num_resolutions - 1)

        layers = [nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1),
                  ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=0.0),
                  AttnBlock(block_in),
                  ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=0.0)
                  ]

        for i in reversed(range(num_resolutions)):
            block_out = ch_mult[i]
            for i_block in range(num_res_blocks+1):
                layers.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=0.0))
                block_in = block_out
                if curr_res in attn_resolutions:
                    layers.append(AttnBlock(block_in))
            if i > 0:
                layers.append(Upsample(block_in, with_conv=True))
            curr_res = curr_res * 2

        layers.append(Normalize(block_in))
        layers.append(Swish())
        layers.append(nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# class Decoder(nn.Module):
#     def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
#                  attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
#                  resolution, z_channels, **ignorekwargs):
#         super().__init__()
#         self.temb_ch = 0
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         self.resolution = resolution
#         self.in_channels = in_channels
#
#         block_in = ch*ch_mult[self.num_resolutions-1]
#         curr_res = resolution // 2**(self.num_resolutions-1)
#         self.z_shape = (1,z_channels,curr_res,curr_res)
#
#         # z to block_in
#         self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
#
#         # middle
#         self.mid = nn.Module()
#         self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
#         self.mid.attn_1 = AttnBlock(block_in)
#         self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
#
#         # upsampling
#         self.up = nn.ModuleList()
#         for i_level in reversed(range(self.num_resolutions)):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_out = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks+1):
#                 block.append(ResnetBlock(in_channels=block_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 if curr_res in attn_resolutions:
#                     attn.append(AttnBlock(block_in))
#             up = nn.Module()
#             up.block = block
#             up.attn = attn
#             if i_level != 0:
#                 up.upsample = Upsample(block_in, resamp_with_conv)
#                 curr_res = curr_res * 2
#             self.up.insert(0, up) # prepend to get consistent order
#
#         # end
#         self.norm_out = Normalize(block_in)
#         self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, z):
#         self.last_z_shape = z.shape
#
#         # timestep embedding
#         temb = None
#
#         # z to block_in
#         h = self.conv_in(z)
#
#         # middle
#         h = self.mid.block_1(h, temb)
#         h = self.mid.attn_1(h)
#         h = self.mid.block_2(h, temb)
#
#         # upsampling
#         for i_level in reversed(range(self.num_resolutions)):
#             for i_block in range(self.num_res_blocks+1):
#                 h = self.up[i_level].block[i_block](h, temb)
#                 if len(self.up[i_level].attn) > 0:
#                     h = self.up[i_level].attn[i_block](h)
#             if i_level != 0:
#                 h = self.up[i_level].upsample(h)
#
#         h = self.norm_out(h)
#         h = nonlinearity(h)
#         h = self.conv_out(h)
#         return h


class Codebook(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    def __init__(self, codebook_size, codebook_dim, beta, init_steps=2000, reservoir_size=2e5):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

        self.q_start_collect, self.q_init, self.q_re_end, self.q_re_step = init_steps, init_steps * 3, init_steps * 30, init_steps // 2
        self.q_counter = 0
        self.reservoir_size = int(reservoir_size)
        self.reservoir = None

    def forward(self, z):
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        batch_size = z.size(0)
        z_flattened = z.view(-1, self.codebook_dim)
        if self.training:
            self.q_counter += 1
            # x_flat = x.permute(0, 2, 3, 1).reshape(-1, z.shape(1))
            if self.q_counter > self.q_start_collect:
                z_new = z_flattened.clone().detach().view(batch_size, -1, self.codebook_dim)
                z_new = z_new[:, torch.randperm(z_new.size(1))][:, :10].reshape(-1, self.codebook_dim)
                self.reservoir = z_new if self.reservoir is None else torch.cat([self.reservoir, z_new], dim=0)
                self.reservoir = self.reservoir[torch.randperm(self.reservoir.size(0))[:self.reservoir_size]].detach()
            if self.q_counter < self.q_init:
                z_q = rearrange(z, 'b h w c -> b c h w').contiguous()
                return z_q, z_q.new_tensor(0), None  # z_q, loss, min_encoding_indices
            else:
                # if self.q_counter < self.q_init + self.q_re_end:
                if self.q_init <= self.q_counter < self.q_re_end:
                    if (self.q_counter - self.q_init) % self.q_re_step == 0 or self.q_counter == self.q_init + self.q_re_end - 1:
                        kmeans = KMeans(n_clusters=self.codebook_size)
                        world_size = dist.get_world_size()
                        print("Updating codebook from reservoir.")
                        if world_size > 1:
                            global_reservoir = [torch.zeros_like(self.reservoir) for _ in range(world_size)]
                            dist.all_gather(global_reservoir, self.reservoir.clone())
                            global_reservoir = torch.cat(global_reservoir, dim=0)
                        else:
                            global_reservoir = self.reservoir
                        kmeans.fit_predict(global_reservoir)  # reservoir is 20k encoded latents
                        self.embedding.weight.data = kmeans.centroids.detach()

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices, shape):
        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


if __name__ == '__main__':
    enc = Encoder()
    dec = Decoder()
    print(sum([p.numel() for p in enc.parameters()]))
    print(sum([p.numel() for p in dec.parameters()]))
    x = torch.randn(1, 3, 512, 512)
    res = enc(x)
    print(res.shape)
    res = dec(res)
    print(res.shape)
