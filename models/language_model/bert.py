# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F

from torch import nn
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process
from transformers import BertConfig, BertModel



class BERT(nn.Module):
    def __init__(self, name: str, train_bert: bool, hidden_dim: int, max_len: int, enc_num):
        super().__init__()
        if name == 'bert-base-uncased':
            self.num_channels = 768
        else:
            self.num_channels = 1024
        self.enc_num = enc_num

        self.bert = BertModel.from_pretrained(name)

        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        if self.enc_num > 0:
            all_encoder_layers, _ = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask)
            # use the output of the X-th transformer encoder layers
            xs = all_encoder_layers[self.enc_num - 1]
        else:
            xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)

        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out

class CustomBertModelLayers(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_channels = 768
        self.load_state_dict(BertModel.from_pretrained("bert-base-uncased").state_dict())

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                layer_idx=None, input_hidden_states=None):

        if layer_idx is None:
            input_shape = input_ids.size()
            seq_length = input_shape[1]
            device = input_ids.device

            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
            hidden_states = self.embeddings(
                input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
            )
            return hidden_states, extended_attention_mask, head_mask
        else:
            layer_module = self.encoder.layer[layer_idx]
            hidden_states = layer_module(input_hidden_states, attention_mask=attention_mask, head_mask=head_mask)[0]
            return hidden_states

def build_bert(args):
    bert = CustomBertModelLayers(config = BertConfig.from_pretrained("bert-base-uncased"))
    return bert