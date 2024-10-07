"""
Basic MobileViT-Track model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.mobilevit_track.layers.conv_layer import Conv2d
from .layers.neck import build_neck, build_feature_fusor
from .layers.head import build_box_head

from lib.models.mobilevit_track.mobilevit_v2 import MobileViTv2_backbone
from lib.utils.box_ops import box_xyxy_to_cxcywh
from easydict import EasyDict as edict

from Wymodelgetter.ops import LowFormerBlock



class LowFormerNeck(nn.Module):

    def __init__(self, lowformit=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        if lowformit:
            self.ff = LowFormerBlock(in_channels=480,fuseconv=True, bb_convattention=True, bb_convin2=True, head_dim_mul=True)

    
    def forward(self, features):
        # print([(key, value.shape) for key, value in features.items()])
        
        lowlevel_up = self.upsample(features[4])
        highlow = torch.cat([lowlevel_up,features[3]],dim=1)
        merged_features = self.ff(highlow)
        
        return merged_features
    
class LowFormer_Track(nn.Module):
    """ This is the base class for MobileViTv2-Track """

    def __init__(self, backbone, box_head, aux_loss=False, head_type="CORNER", feat_fusion=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        
        self.neck = nn.Identity()        
        if feat_fusion:
            self.neck = LowFormerNeck(lowformit=True)
        
        self.box_head = box_head

        
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        
        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, merged_image: torch.Tensor):
        # Backbone
        features = self.backbone(merged_image) # torch.Size([128, 128, 24, 16])

        # Neck
        features = self.neck(features)

        # Forward head
        search_ind = int((features.shape[2]/3)*2)
        out = self.forward_head(features[:,:,:search_ind,:], None)

        return out

    def forward_head(self, backbone_feature, gt_score_map=None):
        """
        backbone_feature: output embeddings of the backbone for search region
        """
        opt_feat = backbone_feature.contiguous()
        bs, _, _, _ = opt_feat.size()
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # bs, Nq, C, HW = opt.size()
        # opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif "CENTER" in self.head_type:
            # run the center head
            score_map_ctr, bbox, size_map, offset_map, max_score = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            # print(outputs_coord_new.shape, score_map_ctr.shape, size_map.shape, offset_map.shape)
            # torch.Size([128, 1, 4]) torch.Size([128, 1, 16, 16]) torch.Size([128, 2, 16, 16]) torch.Size([128, 2, 16,16]
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,
                   "max_score": max_score}
            return out
        else:
            raise NotImplementedError


from Wymodelgetter.get_model import get_lowformer
def build_lowformer_backbone(type, cfg):
    return get_lowformer(config_path="Wymodelgetter/configs/"+type+".yaml", checkpoint_path="Wymodelgetter/checkpoints/"+type+"/evalmodel.pt", cfg=cfg)
    

def show_params_flops(model, cfg):
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        tmpsize = cfg.DATA.TEMPLATE.SIZE
        searsize = cfg.DATA.SEARCH.SIZE
        inp_size = (3, tmpsize+searsize, searsize)
        print("Testing for total size of ",inp_size)
        macs, params = get_model_complexity_info(model, inp_size, as_strings=False,
                                    print_per_layer_stat=False, verbose=False)
        print("MMACS: %d  |  PARAMETERS (M): %.2f" % (macs/1_000_000, params/1_000_000))

def build_lowformer_track(cfg, settings=None, training=True):
    backbone = build_lowformer_backbone(type=cfg.MODEL.BACKBONE.TYPE, cfg=cfg)

    box_head = build_box_head(cfg, cfg.MODEL.HEAD.NUM_CHANNELS)

    model = LowFormer_Track(backbone=backbone, box_head=box_head, aux_loss=False, head_type=cfg.MODEL.HEAD.TYPE, feat_fusion=cfg.MODEL.LOW_FEAT_FUSE)

    from tracking.myutils import to_file
    to_file(str(model),"modelprint.txt")
    # print("Model is printed!")
    # Load Weights
    if 'lowformer_track' in cfg.MODEL.PRETRAIN_FILE and training:
        pass
        # load tracking checkpoint!
    
    show_params_flops(model, cfg)
    
    return model
     



def create_mobilevitv2_backbone(pretrained, width_multiplier, has_mixed_attn, training=False):
    """
    function to create an instance of MobileViT backbone
    Args:
        pretrained:  str
        path to the pretrained image classification model to initialize the weights.
        If empty, the weights are randomly initialized
    Returns:
        model: nn.Module
        An object of Pytorch's nn.Module with MobileViT backbone (i.e., layer-1 to layer-4)
    """
    opts = {}
    opts['mode'] = width_multiplier
    opts['head_dim'] = None
    opts['number_heads'] = 4
    opts['conv_layer_normalization_name'] = 'batch_norm'
    opts['conv_layer_activation_name'] = 'relu'
    opts['mixed_attn'] = has_mixed_attn
    opts["training"] = training
    model = MobileViTv2_backbone(opts)

    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

        assert missing_keys == [], "The backbone layers do not exactly match with the checkpoint state dictionaries. " \
                                   "Please have a look at what those missing keys are!"

        print('Load pretrained model from: ' + pretrained)

    return model