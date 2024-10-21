import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_model.detr import build_detr
from .language_model.bert import build_bert
from .vl_transformer import build_vl_transformer, Triple_Alignment
from .visual_model.transformer import VisionLanguageDecoder

class SegVG(nn.Module):
    def __init__(self, args):
        super(SegVG, self).__init__()
        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len

        ## Extraction ##
        self.visumodel = build_detr(args)
        self.textmodel = build_bert(args)
        self.N = 1 + 5
        self.reg_seg_src = nn.Embedding(self.N, hidden_dim)
        self.reg_seg_pos = nn.Embedding(self.N, hidden_dim)
        self.deepfuse = Triple_Alignment(num_layers=6, embed_dim=hidden_dim, num_heads=8) # this is the triple alignment module, somehow I named it deepfuse

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(768, hidden_dim)
        self.object_proj = nn.Linear(hidden_dim, hidden_dim)

        ## Encoder ##
        num_total = self.num_visu_token + self.num_text_token
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.vl_transformer = build_vl_transformer(args) # this is the encoder module, somehow I named it vl_transformer

        ## Decoder ##
        self.vl_decoder = VisionLanguageDecoder(num_decoder_layers=6, d_model=hidden_dim, nhead=8) # this is the decoder module, somehow I named it vl_decoder

        ## Prediction ##
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.seg_embed = MLP(hidden_dim*2, hidden_dim, 1, 3)

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]

        ## Extraction ## 
        # visual resnet encode #
        visual_src, visual_mask, visual_pos = self.visumodel(samples=img_data)

        # language encode #
        text_src, extended_attention_mask, head_mask = self.textmodel(input_ids=text_data.tensors, attention_mask=text_data.mask)
        text_mask = ~text_data.mask.to(torch.bool)

        # triple alignment #
        tgt_src = self.reg_seg_src.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt_pos = self.reg_seg_pos.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(0, 12, 2):
            # detr
            visual_src, visual_mask, visual_pos = self.visumodel(src=visual_src, mask = visual_mask, pos=visual_pos, layer_idx=i // 2)
            # bert1
            text_src = self.textmodel(input_hidden_states=text_src, attention_mask=extended_attention_mask, head_mask=head_mask[i], layer_idx=i)
            # bert2
            text_src = self.textmodel(input_hidden_states=text_src, attention_mask=extended_attention_mask, head_mask=head_mask[i+1], layer_idx=i+1)
            # triple_align
            visual_src, text_src, tgt_src = self.deepfuse(
                tgt_src = tgt_src, tgt_pos = tgt_pos,
                visual_src=visual_src, visual_mask = visual_mask, visual_pos = visual_pos, 
                text_src=text_src.permute(1, 0, 2), text_mask = text_mask, layer_idx=i // 2)
            text_src = text_src.permute(1, 0, 2) # bert format

        visual_src = self.visu_proj(visual_src)
        text_src = self.text_proj(text_src) 
        text_src = text_src.permute(1, 0, 2) # permute BxLenxC to LenxBxC
        text_mask = text_mask.flatten(1)
        tgt_src = self.object_proj(tgt_src)

        ## Encoder ##
        vl_src = torch.cat([text_src, visual_src], dim=0)
        vl_mask = torch.cat([text_mask, visual_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos) # (L + V)xBxC

        ## Decoder ##
        tgt_src = self.vl_decoder(tgt_src, vg_hs, 
                memory_key_padding_mask = vl_mask, 
                    pos = vl_pos, query_pos = tgt_pos) # L,N,B,C
        pred_box = self.bbox_embed(tgt_src[:, 0]).sigmoid()

        if self.training:
            visua = vg_hs[-400:]
            V,B,C = visua.shape
            L = tgt_src.shape[0]
            N_s = self.N - 1
            visua = visua.view(1,1,V,B,C).expand(L,N_s,V,B,C)
            # L,N,1,B,C & L,N,V,B,C
            visua = torch.cat([
                tgt_src[:, -N_s:].view(L,N_s,1,B,C).expand(L,N_s,V,B,C), 
                visua], dim=-1)
            # L,N,V,B,C --> L,N,V,B,1 --> L,N,V,B --> L,N,B,V
            pred_mask_list = self.seg_embed(visua)[:,:,:,:,0].permute(0,1,3,2)

            return pred_box, pred_mask_list
        else:
            # confidence score
            visua = vg_hs[-400:]
            V,B,C = visua.shape
            
            seg_index = 1
            visua = torch.cat([
                tgt_src[-1, [seg_index]].expand(V,B,C), 
                visua
                ], dim=-1)
            seg_output = self.seg_embed(visua)[:,:,0].transpose(0, 1)

            return pred_box, seg_output.sigmoid()


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