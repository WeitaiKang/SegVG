# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
from torch.nn.parameter import Parameter
from timm.models.layers import trunc_normal_
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

class VisionLanguageEncoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        return self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_vl_transformer(args):
    return VisionLanguageEncoder(
        d_model=args.vl_hidden_dim,
        dropout=args.vl_dropout,
        nhead=args.vl_nheads,
        dim_feedforward=args.vl_dim_feedforward,
        num_encoder_layers=args.vl_enc_layers,
        normalize_before=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.d_h = embed_dim // num_heads
        self.h = num_heads

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = torch.nn.modules.linear.NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, query, key, value, attn_mask=None,  key_padding_mask=None,
                need_weights=True, average_weights=True):
        tgt_len, bs, _ = query.shape
        src_len, bs, _ = key.shape

        # linear projection
        w_q, w_k, w_v = [self.in_proj_weight[i * self.embed_dim : (i+1) * self.embed_dim] for i in range(3)]
        b_q, b_k, b_v = [self.in_proj_bias[i * self.embed_dim : (i+1) * self.embed_dim] for i in range(3)]
        q = torch.nn.functional.linear(query, w_q, b_q)
        k = torch.nn.functional.linear(key, w_k, b_k)
        v = torch.nn.functional.linear(value, w_v, b_v)

        # multi head
        q = q.contiguous().view(tgt_len, bs * self.h, self.d_h).transpose(0, 1)
        k = k.contiguous().view(src_len, bs * self.h, self.d_h).transpose(0, 1)
        v = v.contiguous().view(src_len, bs * self.h, self.d_h).transpose(0, 1)

        # add mask
        key_padding_mask = key_padding_mask.view(bs, 1, 1, src_len). \
            expand(-1, self.h, -1, -1).reshape(bs * self.h, 1, src_len)
        attn_mask_ = key_padding_mask
        # convert mask to float
        new_attn_mask = torch.zeros_like(attn_mask_, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask_, float("-inf")) # Fills elements of self tensor with -inf where attn_mask_ is True
        attn_mask_ = new_attn_mask

        # attn
        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)
        attn_output_weights = torch.baddbmm(attn_mask_, q_scaled, k.transpose(-2, -1))

        # DropKey
        mask_ratio = 0.1
        m_r = torch.ones_like(attn_output_weights) * mask_ratio
        attn_output_weights = attn_output_weights + torch.bernoulli(m_r) * -1e-12
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        attn_output = torch.bmm(attn_output_weights, v) # (bs*h), L, embed/h
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bs, self.embed_dim) # (bs*h), L, embed/h -> L, (bs*h), embed/h -> L*bs, embed
        attn_output = torch.nn.functional.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(tgt_len, bs, self.embed_dim)# L*bs, embed -> L, bs, embed

        # output
        if need_weights:
            attn_output_weights = attn_output_weights.view(bs, self.h, tgt_len, src_len)

            if average_weights:
                attn_output_weights = attn_output_weights.sum(dim=1) / self.h

            return attn_output, attn_output_weights
        else:
            return attn_output



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class Triple_Alignment(nn.Module):

    def __init__(self, num_layers, embed_dim, num_heads):
        super().__init__()
        self.layers = _get_clones(Triple_Alignment_MHA(embed_dim, num_heads), num_layers)

    def forward(self, 
                tgt_src = None, tgt_pos = None,
                visual_src=None, visual_mask: Optional[Tensor] = None, visual_pos: Optional[Tensor] = None, 
                text_src=None, text_mask: Optional[Tensor] = None, layer_idx=0):
        
        layer = self.layers[layer_idx]
        visual_output, text_output, tgt_src = \
            layer(tgt_src=tgt_src, tgt_pos=tgt_pos, 
                  visual_src=visual_src, visual_mask = visual_mask, visual_pos = visual_pos, 
                  text_src=text_src, text_mask = text_mask, layer_idx=layer_idx)
        return visual_output, text_output, tgt_src

class Triple_Alignment_MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(Triple_Alignment_MHA, self).__init__()
        self.embed_dim = embed_dim
        self.d_h = embed_dim // num_heads
        self.h = num_heads
        self.d_h_t = 768 // num_heads

        self.norm_o = nn.LayerNorm(embed_dim)
        self.in_object_q = nn.Linear(embed_dim, embed_dim)
        self.in_object_k = nn.Linear(embed_dim, embed_dim)
        self.in_object_v = nn.Linear(embed_dim, embed_dim)
        self.dropout_o = nn.Dropout(dropout)

        self.norm_v = nn.LayerNorm(embed_dim)
        self.in_visual_q = nn.Linear(embed_dim, embed_dim)
        self.in_visual_k = nn.Linear(embed_dim, embed_dim)
        self.in_visual_v = nn.Linear(embed_dim, embed_dim)
        self.dropout_v = nn.Dropout(dropout)

        self.norm_t = nn.LayerNorm(768)
        self.in_text_q = nn.Linear(768, embed_dim)
        self.in_text_k = nn.Linear(768, embed_dim)
        self.in_text_v = nn.Linear(768, embed_dim)
        self.dropout_t = nn.Dropout(dropout)

        self.out_object = nn.Linear(embed_dim, embed_dim)
        self.out_visual = nn.Linear(embed_dim, embed_dim)
        self.out_text = nn.Linear(embed_dim, 768)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, tgt_src=None, tgt_pos=None, 
                visual_src=None, visual_mask: Optional[Tensor] = None, visual_pos: Optional[Tensor] = None, 
                text_src=None, text_mask: Optional[Tensor] = None, layer_idx=None):
        
        o_len, bs, _ = tgt_src.shape
        v_len, bs, _ = visual_src.shape
        t_len, bs, _ = text_src.shape
        object_src2 = self.norm_o(tgt_src)
        visual_src2 = self.norm_v(visual_src)
        text_src2 = self.norm_t(text_src)

        # linear projection
        object_q = self.in_object_q(self.with_pos_embed(object_src2, tgt_pos))
        object_k = self.in_object_k(self.with_pos_embed(object_src2, tgt_pos))
        object_v = self.in_object_v(object_src2)
        visual_q = self.in_visual_q(self.with_pos_embed(visual_src2, visual_pos))
        visual_k = self.in_visual_k(self.with_pos_embed(visual_src2, visual_pos))
        visual_v = self.in_visual_v(visual_src2)
        text_q = self.in_text_q(text_src2)
        text_k = self.in_text_k(text_src2)
        text_v = self.in_text_v(text_src2)
        otv_q = torch.cat([object_q, text_q, visual_q], dim = 0) # (N+L+V), bs, C
        otv_k = torch.cat([object_k, text_k, visual_k], dim = 0) # (N+L+V), bs, C
        otv_v = torch.cat([object_v, text_v, visual_v], dim = 0) # (N+L+V), bs, C
        otv_len = o_len + t_len + v_len

        # multi head
        otv_q = otv_q.contiguous().view(otv_len, bs * self.h, self.d_h).transpose(0, 1) # bs*h, (N+L+V), C_h
        otv_k = otv_k.contiguous().view(otv_len, bs * self.h, self.d_h).transpose(0, 1)
        otv_v = otv_v.contiguous().view(otv_len, bs * self.h, self.d_h).transpose(0, 1)

        # add mask
        object_mask = torch.zeros(bs*self.h, 1, o_len, dtype = torch.bool).to(visual_mask.device)
        visual_mask = visual_mask.view(bs, 1, 1, v_len). \
            expand(-1, self.h, -1, -1).reshape(bs * self.h, 1, v_len)
        text_mask = text_mask.view(bs, 1, 1, t_len). \
            expand(-1, self.h, -1, -1).reshape(bs * self.h, 1, t_len)
        key_padding_mask = torch.cat([object_mask, text_mask, visual_mask], dim=-1)  # bs*h, 1, (N+L+V)
        attn_mask = torch.zeros_like(key_padding_mask, dtype=otv_q.dtype)
        attn_mask.masked_fill_(key_padding_mask, float("-inf")) # Fills elements of self tensor with -inf where key_padding_mask is True
        # attn
        attn = torch.baddbmm(attn_mask, 
                otv_q / math.sqrt(self.d_h), otv_k.transpose(-2, -1))  # bs*h, (L+V), (L+V)

        # DropKey
        mask_ratio = 0.1
        m_r = torch.ones_like(attn) * mask_ratio
        attn = attn + torch.bernoulli(m_r) * -1e-12
        attn = F.softmax(attn, dim = -1)
        # (bs*h), Len, embed/h -> Len, (bs*h), embed/h -> Len, bs, embed
        attn_output = torch.bmm(attn, otv_v).transpose(0, 1).contiguous().view(otv_len, bs, self.embed_dim)

        o_out = self.dropout_o(self.out_object(attn_output[:o_len])) + tgt_src
        v_out = self.dropout_v(self.out_visual(attn_output[-v_len:])) + visual_src
        t_out = self.dropout_t(self.out_text(attn_output[o_len:-v_len])) + text_src

        return v_out, t_out, o_out