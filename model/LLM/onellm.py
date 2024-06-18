# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math
import functools
import copy

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)
from ..components import RMSNorm
from flash_attn import flash_attn_func

import open_clip


default_linear_init = nn.init.xavier_uniform_


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim -
             1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=default_linear_init,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=default_linear_init,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=default_linear_init,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=default_linear_init,
        )

        self.flash = True
        self.k_cache, self.v_cache = None, None

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.k_cache is None or self.v_cache is None:
            keys, values = xk, xv
        else:
            self.k_cache = self.k_cache.to(xk)
            self.v_cache = self.v_cache.to(xv)
            self.k_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xk
            self.v_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xv
            keys = self.k_cache[:bsz, :start_pos + seqlen]
            values = self.v_cache[:bsz, :start_pos + seqlen]

        output = flash_attn_func(
            xq, keys, values, dropout_p=0.0, causal=mask is not None)
        output = output.contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

    def allocate_kv_cache(self, max_batch_size: int, max_seq_len: int) -> None:
        kv_cache_shape = (max_batch_size, max_seq_len,
                          self.n_local_heads, self.head_dim)
        if self.k_cache is None or self.k_cache.size() != kv_cache_shape:
            self.k_cache = torch.empty(kv_cache_shape)
        if self.v_cache is None or self.v_cache.size() != kv_cache_shape:
            self.v_cache = torch.empty(kv_cache_shape)

    def destroy_kv_cache(self) -> None:
        self.k_cache, self.v_cache = None, None


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * \
            ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=default_linear_init,
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=default_linear_init
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=default_linear_init
        )

    def _silu_gating(self, x, y):
        return F.silu(x) * y

    def forward(self, x):
        return self.w2(self._silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def _forward_ffn(self, h):
        return h + self.feed_forward(self.ffn_norm(h))

    def _forward_attention(self, x, start_pos, freqs_cis, mask, prompt):
        return x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, prompt)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt=None):
        h = self._forward_attention(x, start_pos, freqs_cis, mask, prompt)
        out = self._forward_ffn(h)
        return out


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=nn.init.normal_,
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=default_linear_init,
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        # load clip
        self.clip, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai')
        for param in self.clip.parameters():
            param.requires_grad = False
            param.data = param.data.half()
        self.clip.transformer = None

        self.image_words = 30
        self.cache_image_words = 0  # for inference

        clip_width = self.clip.visual.conv1.out_channels
        # create modal shared modules
        self.resample_layers = nn.ModuleDict()
        self.num_experts = 3
        self.num_resample_layers = 8
        for expert in range(self.num_experts):
            expert = str(expert)
            self.resample_layers[expert] = nn.ModuleList()
            resampler_params = copy.deepcopy(params)
            resampler_params.n_heads = 16
            resampler_params.dim = clip_width
            for layer_id in range(self.num_resample_layers):
                self.resample_layers[expert].append(
                    TransformerBlock(layer_id, resampler_params))

        self.conv1 = nn.ModuleDict()
        self.positional_embedding = nn.ParameterDict()
        self.resample_tokens = nn.ParameterDict()
        self.clip_proj1 = nn.ModuleDict()
        self.clip_proj2 = nn.ModuleDict()
        self.routers = nn.ModuleDict()
        self.start_tag = nn.ParameterDict()
        self.end_tag = nn.ParameterDict()
        self.modals = ['image', 'video', 'audio', 'point', 'rgbd', 'rgbn', 'fmri', 'imu']
        for modal in self.modals:
            if modal in ['image', 'video', 'rgbd', 'rgbn']:
                modal_tokens = 256 + 1
                pass
            elif modal == 'audio':
                self.conv1[modal] = nn.Conv2d(
                    1, clip_width, kernel_size=(16, 16), stride=(10, 10))
                modal_tokens = 1212 + 1
                self.positional_embedding[modal] = nn.Parameter(
                    torch.empty([modal_tokens, clip_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)
            elif modal == 'point':
                from model.lib.point_utils import PointPatchEmbed
                self.conv1[modal] = PointPatchEmbed(
                    in_channels=6, channels=clip_width)
                modal_tokens = 1024 + 1
                self.positional_embedding[modal] = nn.Parameter(
                    torch.empty([modal_tokens, clip_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)
            elif modal == 'fmri':
                self.conv1[modal] = nn.Linear(15724, 8192)
                self.positional_embedding[modal] = nn.Parameter(
                    torch.empty([8+1, clip_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)
            elif modal == 'imu':
                self.conv1[modal] = nn.Conv1d(
                    in_channels=6, out_channels=clip_width, kernel_size=10, bias=False)
                self.positional_embedding[modal] = nn.Parameter(
                    torch.empty([391+1, clip_width]))
                nn.init.normal_(self.positional_embedding[modal], std=0.02)

            self.routers[modal] = Mlp(
                clip_width, clip_width * 4, self.num_experts)

            self.resample_tokens[modal] = nn.Parameter(
                torch.empty([1, 30, resampler_params.dim]))
            nn.init.normal_(self.resample_tokens[modal], std=0.02)

            self.clip_proj1[modal] = nn.Sequential(
                nn.Linear(clip_width, resampler_params.dim),
                nn.LayerNorm(resampler_params.dim))

            self.clip_proj2[modal] = nn.Sequential(
                nn.Linear(resampler_params.dim, params.dim),
                nn.LayerNorm(params.dim))

            self.start_tag[modal] = nn.Parameter(torch.rand(1, 1, params.dim))
            self.end_tag[modal] = nn.Parameter(torch.rand(1, 1, params.dim))
        # TODO: Freeze some parameters at here. Freeze LLM for pretraining and Projection for finetuining.

    # @torch.no_grad()

    def clip_encode_image(self, x, modal='image'):
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                      x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        # use pretrained pos embeding for rest modalities
        pos_embedding = self.clip.visual.positional_embedding
        if modal in ['audio', 'point', 'fmri', 'imu']:
            pos_embedding = self.positional_embedding[modal]

        x = x + pos_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        # if self.clip.visual.proj is not None:
        #    x = x @ self.clip.visual.proj

        return x

    def encode_image(self, x, modal='image'):
        bsz = x.size(0)
        T = 1
        if modal in ['image']:
            # modified from CLIP
            x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        elif modal in ['audio', 'imu']:
            x = self.conv1[modal](x)
        elif modal == 'point':
            # [B, 16384, 6] -> [B, 1024, 1024, 1]
            x = self.conv1[modal](x.float()).to(x.dtype)
        elif modal in ['video', 'rgbd', 'rgbn']:
            # [B, 15, 3, 224, 224]
            B, T = x.shape[:2]
            bsz = B * T
            x = x.reshape(bsz, *x.shape[2:])
            x = self.clip.visual.conv1(x)
        elif modal == 'fmri':
            x = self.conv1[modal](x)
            # [B, 1, 8196] -> [B, 1024, 8]
            x = x.reshape(x.size(0), self.clip.visual.conv1.out_channels, -1)

        image_feats = self.clip_encode_image(x, modal=modal)
        # take mean on time dimension
        # all inputs are reduced to [B, L, D]
        bsz = int(bsz / T)
        image_feats = image_feats.reshape(
            bsz, T, *image_feats.shape[1:]).mean(dim=1)

        image_feats = self.clip_proj1[modal](image_feats)
        image_feats = torch.cat(
            [self.resample_tokens[modal].repeat(bsz, 1, 1), image_feats], dim=1)

        # routing modalites
        # [B, L, D]->[B, L, N]
        routing_weights = self.routers[modal](image_feats).sigmoid()
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        image_feats_experts = []
        for expert_id in range(self.num_experts):
            image_feats_expert = image_feats
            for layer in self.resample_layers[str(expert_id)]:
                image_feats_expert = layer(image_feats_expert, 0, None, None)

            image_feats_expert = image_feats_expert[:, :self.resample_tokens[modal].size(1)]
            routing_weight = routing_weights[:, :self.resample_tokens[modal].size(
                1), expert_id]
            # [B, L, D] * [B, L, 1]
            image_feats_expert = image_feats_expert * routing_weight[:, :, None]

            image_feats_experts.append(image_feats_expert)

        image_feats = sum(image_feats_experts)
        image_feats = self.clip_proj2[modal](image_feats)

        return image_feats

    def forward(self, examples, image=None, modal='image'):
        self._destroy_kv_cache()  # training always disables kv cache
        modal = modal[0]
        _bsz, seqlen = examples.shape
        h = self.tok_embeddings(examples)
        self.freqs_cis = self.freqs_cis.to(h.device)

        start_pos = 0
        prefix_len = 0
        if image is not None:
            h_bos, h_caption = h[:, :1], h[:, 1:]
            image_tokens = self.encode_image(image, modal)
            h = torch.cat((h_bos, self.start_tag[modal].expand(
                _bsz, -1, -1), image_tokens, self.end_tag[modal].expand(_bsz, -1, -1), h_caption), dim=1)
            # bos + image token + start_tag[modal], end_tag[modal] is used for caption generation
            prefix_len = image_tokens.shape[1] + 1 + 1
            seqlen = h.shape[1]

        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, prefix_len:, :])
        return output

    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int, image=None, modal='image'):
        modal = modal[0] if isinstance(modal, list) else modal
        _bsz, seqlen = tokens.shape
        if start_pos == 0:
            # kv cache will not re-allocate if size is unchanged
            self._allocate_kv_cache(_bsz)
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)

        if image is not None:
            h_bos, h_caption = h[:, :1], h[:, 1:]
            image_tokens = self.encode_image(image, modal)
            self.cache_image_words = image_tokens.shape[1]
            h = torch.cat((h_bos, self.start_tag[modal].repeat(_bsz, 1, 1), image_tokens, self.end_tag[modal].repeat(_bsz, 1, 1), h_caption), dim=1)
            seqlen = h.shape[1]
            freqs_cis = self.freqs_cis[0: seqlen]
        else:
            if start_pos == 0:
                self.cache_image_words = 0
                freqs_cis = self.freqs_cis[0: seqlen]
            else:
                # if image was not None when start_pos=0,
                # the offset should be added to start_pos within later forward_inference calls
                start_pos = start_pos + self.cache_image_words
                freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        # freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()

    def _allocate_kv_cache(self, max_batch_size: int) -> None:
        for layer in self.layers:
            layer.attention.allocate_kv_cache(
                max_batch_size, self.params.max_seq_len)

    def _destroy_kv_cache(self) -> None:
        for layer in self.layers:
            layer.attention.destroy_kv_cache()
