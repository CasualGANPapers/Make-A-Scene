"""
taken from: https://github.com/ai-forever/ru-dalle/blob/master/rudalle/dalle/transformer.py slightly modified
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self,
                 hidden_dim,
                 num_attn_heads,
                 attn_dropout_prob,
                 out_dropout_prob,
                 cogview_pb_relax=True,
                 rudalle_relax=False
                 ):
        super(SelfAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_attn_heads = num_attn_heads
        self.d = math.sqrt(self.hidden_dim // self.num_attn_heads)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.attn_drop = nn.Dropout(attn_dropout_prob)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_drop = nn.Dropout(out_dropout_prob)

        self.cogview_pb_relax = cogview_pb_relax
        self.rudalle_relax = rudalle_relax

    def split_heads(self, x):
        new_shape = x.size()[:-1] + [self.num_attn_heads, self.hidden_dim // self.num_attn_heads]
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def calculate_attention(self, q, k, mask):
        k_t = k.transpose(-1, -2)
        mask_value = 10000.
        if self.cogview_pb_relax:
            if self.rudalle_relax:
                sigma = k_t.std()
                attn_scores = torch.matmul(q / d, k_t / sigma)
                attn_scores_max = attn_scores.detach().max(dim=-1)[0]
                attn_scores_min = (attn_scores.detach() + 65504).min(dim=-1)[0]
                shift = torch.min(attn_scores_min, attn_scores_max).unsqueeze(-1).expand_as(attn_scores) / 2
                attn_scores = (attn_scores - shift) / sigma
                mask_value = 65504
            else:
                attn_scores = torch.matmul(q / d, k_t)
        else:
            attn_scores = torch.matmul(q, k_t) / d

        mask = mask[:, :, -attn_scores.shape[-2]:]
        attn_scores = mask * attn_scores - (1. - mask) * mask_value
        if self.cogview_pb_relax and not self.rudalle_relax:
            alpha = 32
            attn_scores_scaled = attn_scores / alpha
            attn_scores_scaled_max, _ = attn_scores_scaled.detach().view(
                [attn_scores.shape[0], attn_scores.shape[1], -1]).max(dim=-1)
            attn_scores_scaled_max = attn_scores_scaled_max[..., None, None].expand(
                [-1, -1, attn_scores.size(2), attn_scores.size(3)])
            attn_scores = (attn_scores_scaled - attn_scores_scaled_max) * alpha
        return attn_scores

    def forward(self, x, mask, use_cache=False, cache=None):
        if use_cache and cache is not None:
            qkv = self.qkv(x[:, cache[0].shape[-2]:, :])
        else:
            qkv = self.qkv(x)

        q, k, v = torch.split(qkv, qkv.shape[-1] // 3, dim=-1)  # probably use different dim
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

        if use_cache and cache is not None:
            past_k, past_v, past_output = cache
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
            attn_scores = self.calculate_attention(q, k, mask)
        else:
            attn_scores = self.calculate_attention(q, k, mask)

        attn_probs = nn.Softmax(dim=-1)(attn_scores)  # [b, np, s, s]
        attn_probs = self.attn_drop(attn_probs)

        if self.rudalle_relax:
            scale = v.detach().max().item()
            context = torch.matmul(attn_probs, v / scale)
        else:
            context = torch.matmul(attn_probs, v)

        context = context.permute(0, 2, 1, 3).view(context.shape[0], context.shape[1],
                                                   context.shape[2] * context.shape[3])

        if self.rudalle_relax:
            scale = context.detach().max().item()
            context /= scale

        out = self.out_proj(context)
        if use_cache and cache is not None:
            out = torch.concat([past_output, out], dim=-2)

        if use_cache:
            cache = k, v, out

        out = self.out_drop(out)
        return out, cache


class MLP(nn.Module):
    def __init__(self,
                 hidden_dim,
                 dropout_prob,
                 rudalle_relax=False
                 ):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.lin2 = nn.Linear(4 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.rudalle_relax = rudalle_relax

    def forward(self, x):
        x = self.lin1(x)
        x = gelu(x)
        if self.rudalle_relax:
            scale = x.detach().max().item() / 4
            x = self.lin2(x / scale)
            x = (x / x.detach().max(dim=-1)[0].unsqueeze(-1)) * scale
        else:
            x = self.lin2(x)
        return self.dropout(x)


class TransformerLayer(nn.Module):
    def __init__(self,
                 hidden_dim,
                 num_attn_heads,
                 attn_dropout_prop,
                 out_dropout_prob,
                 cogview_pb_relax=True,
                 cogview_sandwich_layernorm=True,
                 cogview_layernorm_prescale=False,
                 rudalle_relax=False
                 ):
        super().__init__()
        self.cogview_pb_relax = cogview_pb_relax
        self.cogview_sandwich_layernorm = cogview_sandwich_layernorm
        self.cogview_layernorm_prescale = cogview_layernorm_prescale
        self.rudalle_relax = rudalle_relax

        self.ln_in = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.ln_out = nn.LayerNorm(hidden_dim, eps=1e-5)
        if cogview_sandwich_layernorm:
            self.first_ln_sandwich = nn.LayerNorm(hidden_dim, eps=1e-5)
            self.second_ln_sandwich = nn.LayerNorm(hidden_dim, eps=1e-5)

        self.attn = SelfAttention(hidden_dim=hidden_dim,
                                  num_attn_heads=num_attn_heads,
                                  attn_dropout_prob=attn_dropout_prop,
                                  out_dropout_prob=out_dropout_prob,
                                  cogview_pb_relax=cogview_pb_relax,
                                  rudalle_relax=rudalle_relax)

        self.mlp = MLP(hidden_dim=config.dim,
                       dropout_prob=config.drop_prob,
                       rudalle_relax=config.rudalle_relax)

    def forward(self, x, mask, cache=None, use_cache=False, mlp_cache=False):
        if self.cogview_layernorm_prescale:
            ln_in = self.ln_in(x / x.detach().max(dim=-1)[0].unsqueeze(-1))
        else:
            ln_in = self.ln_in(x)
        attn_out, new_cache = self.attn(ln_in, mask, cache, use_cache)

        if self.cogview_sandwich_layernorm:
            if self.cogview_layernorm_prescale:
                attn_out = self.first_ln_sandwich(attn_out / attn_out.detach().max(dim=-1)[0].unsqueeze(-1))
            else:
                attn_out = self.first_ln_sandwich(attn_out)

        x = x + attn_out
        cached = 0 if cache is None else cache[0].shape[2]

        if self.cogview_layernorm_prescale:
            ln_out = self.ln_out(x / x.detach().max(dim=-1)[0].unsqueeze(-1))
        else:
            ln_out = self.ln_out(x)

        if use_cache and cached:
            mlp_out = torch.cat(
                (cache[-1] if mlp_cache else ln_out[..., :cached, :], self.mlp(ln_out[..., :cached, :])), dim=-2)
            if mlp_cache:
                new_cache = new_cache + (mlp_out,)
        else:
            mlp_out = self.mlp(ln_out)

        if self.cogview_sandwich_layernorm:
            mlp_out = self.second_ln_sandwich(mlp_out)

        x = x + mlp_out

        return x, new_cache


class Transformer(nn.Module):
    def __init__(self,
                 num_layers,
                 hidden_dim,
                 num_attn_heads,
                 attn_dropout_prop,
                 out_dropout_prob,
                 image_tokens_per_dim=32,
                 text_length=256,
                 cogview_pb_relax=True,
                 cogview_sandwich_layernorm=True,
                 cogview_layernorm_prescale=False,
                 rudalle_relax=False
                 ):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.cogview_pb_relax = cogview_pb_relax
        self.rudalle_relax = rudalle_relax

        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim,
                num_attn_heads,
                attn_dropout_prop,
                out_dropout_prob,
                cogview_pb_relax,
                cogview_sandwich_layernorm,
                cogview_layernorm_prescale,
                rudalle_relax
            ) for _ in num_layers
        ])

        self.register_buffer("mask", self._create_mask(text_length, image_tokens_per_dim))
        self.final_ln = nn.LayerNorm(hidden_dim, eps=1e-5)

    def _create_mask(self, text_length, image_tokens_per_dim=32):
        size = text_length + image_tokens_per_dim ** 2
        return torch.tril(torch.ones(size, size, dtype=torch.float32))

    def get_block_size(self):
        return self.block_size

    def forward(self, x, attn_mask, cache=None, use_cache=None):
        if cache is None:
            cache = {}

        for i, layer in enumerate(self.layers):
            mask = attn_mask
            layer_mask = self.mask[:mask.size(2), :mask.size(3)]
            mask = torch.mul(attention_mask, layer_mask)
            x, layer_cache = layer(x, mask, cache.get(i), mlp_cache=i == len(self.layers) - 1, use_cache=use_cache)
            cache[i] = layer_cache

        if self.rudalle_relax:
            ln_out = self.final_ln(x / x.detach().max(dim=-1)[0].unsqueeze(-1))
        else:
            ln_out = self.final_ln(x)

        return ln_out, cache


class MakeAScene(nn.Module):
    def __init__(self,
                 transformer_config,
                 hidden_dim,
                 image_vocab_size,
                 seg_vocab_size,
                 text_vocab_size,
                 image_tokens_per_dim,
                 text_length,
                 seg_length
                 ):
        super(MakeAScene, self).__init__()
        self.image_length = image_tokens_per_dim ** 2
        self.text_length = text_length
        self.total_length = self.text_length + self.seg_length + self.image_length
        self.text_vocab_size = text_vocab_size

        self.transformer = Transformer(**transformer_config)

        self.image_token_embedding = nn.Embedding(image_vocab_size, hidden_dim)
        self.seg_token_embedding = nn.Embedding(seg_vocab_size, hidden_dim)
        self.text_token_embedding = nn.Embedding(text_vocab_size, hidden_dim)
        self.padding_token_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.text_pos_embeddings = _init_weightstorch.nn.Embedding(text_length + 1, hidden_dim)
        self.image_row_embeddings = torch.nn.Embedding(image_tokens_per_dim, hidden_dim)
        self.image_col_embeddings = torch.nn.Embedding(image_tokens_per_dim, hidden_dim)
        self._init_weights(self.text_pos_embeddings)
        self._init_weights(self.image_row_embeddings)
        self._init_weights(self.image_col_embeddings)

        self.to_logits = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),  # TODO: check if this is redundant
            torch.nn.Linear(hidden_dim, image_vocab_size + seg_vocab_size + text_vocab_size),
        )

        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_image_pos_embeddings(self, image_input_ids, past_length=0):
        input_shape = image_input_ids.size()
        row_ids = torch.arange(past_length, input_shape[-1] + past_length,
                               dtype=torch.long, device=self.device) // self.image_tokens_per_dim
        row_ids = row_ids.unsqueeze(0).view(-1, input_shape[-1])
        col_ids = torch.arange(past_length, input_shape[-1] + past_length,
                               dtype=torch.long, device=self.device) % self.image_tokens_per_dim
        col_ids = col_ids.unsqueeze(0).view(-1, input_shape[-1])
        return self.image_row_embeddings(row_ids) + self.image_col_embeddings(col_ids)

    def forward(self, text_tokens, seg_tokens, img_tokens):
        text_range = torch.arange(self.text_length)
        text_range += (self.text_vocab_size - self.text_length)
        text_range = text_range.to(self.device)
        text_tokens = torch.where(text_tokens == 0, text_range, text_tokens)
        text_pos = self.text_pos_embeddings(torch.arange(text_tokens.shape[1], device=self.device))
        text_embeddings = self.text_token_embedding(text_tokens) + text_pos

        seq_pos = self.seq_pos_embeddings(torch.arange(seg_tokens.shape[1], device=self.device))
        seq_embeddings = self.seg_token_embedding(seg_tokens) + text_pos

        embeddings = torch.cat((text_embeddings, seq_embeddings), dim=1)
        if img_tokens is not None:
            img_pos = self.get_image_pos_embeddings(img_tokens)
            image_embeddings = self.image_embeddings(img_tokens) + img_pos
            embeddings = torch.cat((embeddings, image_embeddings), dim=1)
            
        attention_mask = torch.tril(
            torch.ones((embeddings.shape[0], 1, self.total_length, self.total_length), device=self.device)
        )
        attention_mask = attention_mask[:, :, :embeddings.shape[1], :embeddings.shape[1]]

        transformer_output, present_cache = self.transformer(
            embeddings, attention_mask,
            cache=None, use_cache=False
        )

        logits = self.to_logits(transformer_output)
        return logits[:, -self.image_length:, :]


    # def forward(self, text_tokens, seg_tokens, img_tokens):
    #     if self.training:
    #         mask = torch.bernoulli(self.pkeep * torch.ones_like(img_tokens))
    #         mask = mask.round().to(dtype=torch.int64)
    #         random_tokens = torch.randint_like(img_tokens, self.vocab_size)
    #         random_img_tokens = mask * img_tokens + (1 - mask) * random_tokens
    #     else:
    #         random_img_tokens = img_tokens
    #
    #     tokens = torch.cat([text_tokens, seg_tokens, random_img_tokens], dim=1)
    #     target = img_tokens
    #
    #     logits = self.transformer(tokens[:, :-1])
    #     logits = logits[:, text_tokens.shape[1] + seg_tokens.shape[1] - 1:]
    #
    #     return logits, target



