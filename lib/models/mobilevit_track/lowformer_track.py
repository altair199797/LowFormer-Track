"""
Basic MobileViT-Track model.
"""
import math
import os
from typing import List, Dict

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

    def __init__(self, lowformit=False, add_stage=0,  backbone_arch="b15"):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.add_stage = add_stage
        feat_base = 20 if backbone_arch == "b15" else (32 if backbone_arch=="b3" else (16 if backbone_arch=="b1" else -1 ))
        
        self.downit, self.downit2, self.combineit, self.combineit2 = nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity()
        if add_stage:
            self.downit = nn.Sequential(nn.Conv2d(feat_base*4, feat_base*8,3, stride=2, padding=1))
            self.combineit = nn.Sequential(nn.Conv2d(feat_base*16, feat_base*8, 1, stride=1, padding=0))
            if add_stage == 2:
                self.downit2 = nn.Sequential(nn.Conv2d(feat_base*2, feat_base*4,3, stride=2, padding=1))
                self.combineit2 = nn.Sequential(nn.Conv2d(feat_base*8, feat_base*4, 1, stride=1, padding=0))
                
    
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")  
        if lowformit:
            self.ff = LowFormerBlock(in_channels=feat_base*24,fuseconv=True, bb_convattention=True, bb_convin2=True, head_dim_mul=True)

        
    def forward(self, features: Dict[int, torch.Tensor]): 
        # print([(key, value.shape) for key, value in features.items()])
        # stage1: 40,96,64 |  stage2: 80,48,32 | stage3: 160,24,16 | stage4: 320,12,8
        if self.add_stage:
            if self.add_stage == 2:
                feat_1_down = self.downit2(features[1])
                features[2] = self.combineit2(torch.cat([feat_1_down,features[2]],dim=1))

            feat_2_down = self.downit(features[2]) # 80,48,32 ->  160,24,16
            features[3] = self.combineit(torch.cat([feat_2_down,features[3]],dim=1))

        lowlevel_up = self.upsample(features[4])
        highlow = torch.cat([lowlevel_up,features[3]],dim=1)
        merged_features = self.ff(highlow)
        
        return {4:merged_features}


class LowFormerNeckV2(nn.Module):

    def __init__(self, lowformit=False, add_stage=0, backbone_arch="b15", backbone=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.add_stage = add_stage
        feat_base = 20 if backbone_arch == "b15" else (32 if backbone_arch=="b3" else (16 if backbone_arch=="b1" else -1 ))
        assert add_stage == 2, add_stage
                
        self.downit, self.downit2, self.combineit, self.combineit2 = nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity()
        
        
        self.downit1 = nn.Sequential(nn.Conv2d(feat_base*2, feat_base*4,3, stride=2, padding=1))
        self.downit2 = nn.Sequential(nn.Conv2d(feat_base*8, feat_base*16,3, stride=2, padding=1))
        # self.downit3 = nn.Sequential(nn.Conv2d(feat_base*4, feat_base*8,3, stride=2, padding=1))
        
        self.combineit = nn.Sequential(nn.Conv2d(feat_base*24, feat_base*16, 1, stride=1, padding=0))
        self.combineit2 = nn.Sequential(nn.Conv2d(feat_base*8, feat_base*8, 1, stride=1, padding=0))
        self.combineit3 = nn.Sequential(nn.Conv2d(feat_base*32, feat_base*16, 1, stride=1, padding=0))
        
                
    
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")  
        if lowformit:
            self.ff = LowFormerBlock(in_channels=feat_base*16,fuseconv=True, bb_convattention=True, bb_convin2=True, head_dim_mul=True)
            # print(backbone)
            # for layer in backbone.stages[-1].modules():
            #     print(layer,"\n\n--------------------\n\n")

    # -> [N,320,16,16]
    def forward(self, features: Dict[int, torch.Tensor]): 
        # print([(key, value.shape) for key, value in features.items()])
        # stage1: 40,96,64 |  stage2: 80,48,32 | stage3: 160,24,16 | stage4: 320,12,8
        features[4] = self.upsample(features[4]) # -> [N,320,16,16]
        features[3] = features[3]  # [N,160,16,16]
        lower_ff = self.combineit(torch.cat([features[3], features[4]],dim=1)) # -> [N,320,16,16]
        
        features[1] = self.downit1(features[1]) # ->[N,80,32,32]
        upper_ff = self.combineit2(torch.cat([features[1], features[2]], dim=1)) # -> [N,160,32,32]
        upper_ff = self.downit2(upper_ff) # -> [N,320,16,16]
        
        highlow = self.combineit3(torch.cat([lower_ff, upper_ff],dim=1)) 
        
        # features[2] = self.downit(features[2]) # [N,160,16,16]
        # features[1] = self.downit3(features[1]) # ->[N,160,16,16]
        # highlow = self.combineit(torch.cat([features[1],features[2],features[3],features[4]], dim=1))
        
        merged_features = self.ff(highlow)
        
        return {4:merged_features}
    
    
    
class LowFormer_Track(nn.Module):
    """ This is the base class for MobileViTv2-Track """

    def __init__(self, backbone, box_head, aux_loss=False, head_type="CORNER", feat_fusion=False, cfg=None, additional_layer=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.cfg = cfg 
        self.feat_fusion = feat_fusion
        self.additional_layer = additional_layer
    
        self.backbone = backbone
        
        self.neck = nn.Identity()        
        if feat_fusion:
            if cfg.MODEL.LOW_FEAT_FUSEV4:
                self.neck = LowFormerNeckV2(lowformit=True, add_stage=int(cfg.MODEL.LOW_FEAT_FUSEV2) + int(cfg.MODEL.LOW_FEAT_FUSEV3),backbone_arch=cfg.MODEL.BACKBONE.TYPE, backbone=backbone)
            else:
                self.neck = LowFormerNeck(lowformit=True, add_stage=int(cfg.MODEL.LOW_FEAT_FUSEV2) + int(cfg.MODEL.LOW_FEAT_FUSEV3),backbone_arch=cfg.MODEL.BACKBONE.TYPE)
        
        self.box_head = box_head

        
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        
        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
            
        self.no_grad_backbone = self.cfg.TRAIN.NO_GRAD_BACKBONE
        self.no_template_feats = self.cfg.MODEL.NO_TEMPLATE_FEATS
        self.model_return_template =  self.cfg.MODEL.RETURN_TEMPLATE
        
        if self.additional_layer:
            self.add_layers = nn.ModuleList([LowFormerBlock(in_channels=chs,fuseconv=True, bb_convattention=True, bb_convin2=True, head_dim_mul=True) for chs in [80,160,320]])

    def execute_backbone(self, merged_image: torch.Tensor):
        if self.additional_layer:
            features = {}
            merged_image = self.backbone.input_stem(merged_image)
            merged_image = self.backbone.stages[0](merged_image)
            features[1] = merged_image.clone()
            # print(merged_image.shape)
            
            merged_image = self.backbone.stages[1](merged_image)
            merged_image = self.add_layers[0](merged_image)
            features[2] = merged_image.clone()
            # print(merged_image.shape)
            
            merged_image = self.backbone.stages[2](merged_image)
            merged_image = self.add_layers[1](merged_image)
            features[3] = merged_image.clone()
            # print(merged_image.shape)
            
            merged_image = self.backbone.stages[3](merged_image)
            merged_image = self.add_layers[2](merged_image)
            features[4] = merged_image.clone()
            # print(merged_image.shape)
            
        else:
            features = self.backbone(merged_image) # torch.Size([128, 128, 24, 16])

        return features
        
    def forward(self, merged_image: torch.Tensor):
        ### Backbone
        if self.no_grad_backbone:#self.cfg.TRAIN.NO_GRAD_BACKBONE:
            with torch.no_grad():
                features = self.execute_backbone(merged_image) # torch.Size([128, 128, 24, 16])
        else:
            features = self.execute_backbone(merged_image)


        # cut off template features
        if self.no_template_feats:
            # assert False
            features = {key: value[:,:,:int((value.shape[2]/3)*2),:] for key, value in features.items() }
        
        ### Neck
        features = self.neck(features)
        assert len(list(features.keys())) == 1, features.keys()
        features = features[list(features.keys())[0]]

        # Cut off features before head
        features = features[:,:,:int((features.shape[2]/3)*2),:]
        
        ### Forward head
        if self.model_return_template:
            out = self.forward_head(features)
        else:
            # if not features.shape[2] == features.shape[3]:
            #     search_ind = int((features.shape[2]/3)*2)
            # else:
            #     search_ind = features.shape[2]
            out = self.forward_head(features)

        return out

    def forward_head(self, backbone_feature):
        """
        backbone_feature: output embeddings of the backbone for search region
        """
        opt_feat = backbone_feature.contiguous()
        bs, _, _, _ = opt_feat.size()
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # bs, Nq, C, HW = opt.size()
        # opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            assert False
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif "CENTER" in self.head_type or self.head_type == "STRIDEHEAD":
            
            # run the center head
            # if self.cfg.MODEL.HEAD.TYPE == "APCENTER":# or self.cfg.MODEL.HEAD.TYPE == "STRIDEHEAD":
            #     score_map_ctr, bbox, size_map, offset_map, max_score, grid_map = self.box_head(opt_feat, gt_score_map)
            # else:
            #     score_map_ctr, bbox, size_map, offset_map, max_score = self.box_head(opt_feat, gt_score_map)
            #     grid_map = None
            
            score_map_ctr, bbox, size_map, offset_map, max_score, grid_map = self.box_head(opt_feat) 
            
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
      
            # print(outputs_coord_new.shape, score_map_ctr.shape, size_map.shape, offset_map.shape)
            # torch.Size([128, 1, 4]) torch.Size([128, 1, 16, 16]) torch.Size([128, 2, 16, 16]) torch.Size([128, 2, 16,16]
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,
                   "max_score": max_score,
                   "grid_map": grid_map}
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

    model = LowFormer_Track(backbone=backbone, box_head=box_head, aux_loss=False, head_type=cfg.MODEL.HEAD.TYPE, feat_fusion=cfg.MODEL.LOW_FEAT_FUSE, cfg=cfg, additional_layer=cfg.MODEL.ADD_BACKBONE_LAYER)

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