"""
Basic MobileViT-Track model.
"""
import math
import os
from typing import List

import torch, torchvision
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.mobilevit_track.layers.conv_layer import Conv2d
from .layers.neck import build_neck, build_feature_fusor
from .layers.head import build_box_head

from lib.models.mobilevit_track.mobilevit_v2 import MobileViTv2_backbone
from lib.utils.box_ops import box_xyxy_to_cxcywh
from easydict import EasyDict as edict


class MobileViTv2_Track(nn.Module):
    """ This is the base class for MobileViTv2-Track """

    def __init__(self, backbone, neck, feature_fusor, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        if neck is not None:
            self.neck = neck
            self.feature_fusor = feature_fusor
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor, search: torch.Tensor, template_anno=None):
        # print("sizes:",search.shape, template.shape)
        if template_anno is None:
            x, z = self.backbone(x=search, z=template)
        else:
            x, z = self.backbone(x=search, z=template, template_anno=template_anno)

        # print("after bb:", x.shape, z.shape)
        # Forward neck
        x, z = self.neck(x, z)

        # print("after neck:", x.shape, z.shape)
        # Forward feature fusor
        if self.feature_fusor is not None:
            feat_fused = self.feature_fusor(z, x)
        else:
            feat_fused = (x,z) # actually query+attnmap

        # print("after ff:", x.shape, z.shape)
        # Forward head
        out = self.forward_head(feat_fused, None)
        return out

    def forward_head(self, backbone_feature, gt_score_map=None):
        """
        backbone_feature: output embeddings of the backbone for search region
        """
        if isinstance(backbone_feature, tuple):
            if isinstance(backbone_feature[0], tuple):
                opt_feat = backbone_feature[0][0].contiguous()
            else:
                opt_feat = backbone_feature[0].contiguous()
            bs, _, _ = opt_feat.size()
        
        else:
            opt_feat = backbone_feature.contiguous()
            bs, _, _, _ = opt_feat.size()
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # bs, Nq, C, HW = opt.size()
        # opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "EFFTRACK":
            bbox, scores = self.box_head(*backbone_feature)
            return {'pred_boxes': bbox.view(bs, 1, 4),
                   'scores': scores,
                   } 
        elif self.head_type == "CORNER":
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
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError



class LowFormerWrapper(nn.Module):

    def __init__(self, backbone, training, type_emb=False, type_embv2=False):
        super().__init__()
        self.backbone = backbone
        self.training_it = training
        
        self.conv_1 = self.backbone.input_stem
        self.layer_1 = self.backbone.stages[0]
        self.layer_2 = self.backbone.stages[1]
        
        self.type_emb = type_emb
        self.type_embv2 = type_embv2

        if self.type_emb:
            self.token_type_embed = nn.Parameter(torch.empty(3, 80))
            # torch.nn.init.normal_(self.token_type_embed, std=0.02)
            if type_embv2:
                torch.nn.init.trunc_normal_(self.token_type_embed, mean=1.0, std=0.02, a=-1.0, b=3.0)
            else:
                torch.nn.init.trunc_normal_(self.token_type_embed, std=0.02, a=-2.0, b=2.0)

    def forward(self, x, z, template_anno=None):
        # 224 -> 112
        # print("in model class")
        orig_inp_size = int(x.shape[2]//2)
        x = self.backbone.input_stem(x)
        if self.training_it:
            # 112 -> 56
            z = self.backbone.input_stem(z)
            # print("input stem executed")

        for ind, stage in enumerate(self.backbone.stages):
            if ind == len(self.backbone.stages)-1:
                
                ## Calc masks
                if not template_anno is None and self.type_emb: # template anno in template space, [1,128,4]
                    # x = x + self.token_type_embed[0,:].reshape(1,z.shape[1],1,1)
                    type_mask = torch.zeros(z.shape[0],1,z.shape[2], z.shape[3]) # [128,1, 14, 14]
                    for i in range(x.shape[0]):
                        xcord, ycord, w, h = int(template_anno[0,i,0]*orig_inp_size/8), int(template_anno[0,i,1]*orig_inp_size/8), int(template_anno[0,i,2]*orig_inp_size/8), int(template_anno[0,i,3]*orig_inp_size/8)
                        xcord, ycord = max(0,xcord), max(ycord,0) 
                        w, h = min(xcord+w,14) - xcord, min(ycord+h,14) - ycord
                        temp = torch.full([h,w],1)
                        # print("h:",h,"w:",w,"x:",xcord,"y:",ycord, template_anno[0,i,:])
                        type_mask[i,:,ycord:ycord+h,xcord:xcord+w] = temp
                    type_mask = type_mask.cuda()
                    
                    # z = z + type_mask * self.token_type_embed[1,:].reshape(1,z.shape[1],1,1) + (1 - type_mask) * self.token_type_embed[2,:].reshape(1,z.shape[1],1,1)

                elif self.type_emb:
                    type_mask = torch.zeros(z.shape[0],1,z.shape[2], z.shape[3]).cuda()
                    h,w = z.shape[2], z.shape[3]
                    type_mask[:,:,int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)] = 1

                    # x = x + self.token_type_embed[0,:].reshape(1,z.shape[1],1,1)
                    # z = z + type_mask * self.token_type_embed[1,:].reshape(1,z.shape[1],1,1) + (1 - type_mask) * self.token_type_embed[2,:].reshape(1,z.shape[1],1,1)
                
                ## Actual Adding/Multiplication
                if self.type_embv2:
                    x = x * self.token_type_embed[0,:].reshape(1,z.shape[1],1,1)
                    z = z * (type_mask * self.token_type_embed[1,:].reshape(1,z.shape[1],1,1) + (1 - type_mask) * self.token_type_embed[2,:].reshape(1,z.shape[1],1,1))
                elif self.type_emb:
                    x = x + self.token_type_embed[0,:].reshape(1,z.shape[1],1,1)
                    z = z + type_mask * self.token_type_embed[1,:].reshape(1,z.shape[1],1,1) + (1 - type_mask) * self.token_type_embed[2,:].reshape(1,z.shape[1],1,1)
                    
                    

                merged = torch.cat([x,torch.cat([z,z.clone()],dim=3)], dim=2)
                merged = stage(merged)
                # print("prelast stage first if")
                x, z = merged[:,:,:z.shape[2],:], merged[:,:,z.shape[2]:,:int(z.shape[3]//2)]
            else:
                # print("other if")
                x = stage(x)
                if self.training_it or ind > 1:
                    z = stage(z)
                    # print("z execution")
                # 0: 112 -> 56 , 56 -> 28
                # 1: 56 -> 28,  28 -> 14
                # 2: 28 -> 14 ,  14 -> 7

        return x, z



        


extra_bb = True 
multi_scale = True
two_bbs = True
sep_queries = True
clamp_offset = True
detach_attn = False ##
predict_center = True

class EffTrackWrapper(nn.Module):

    def __init__(self, backbone, training):
        super().__init__()
        self.backbone = backbone
        self.training = training
        if two_bbs:
            import copy
            self.backbone2 = copy.deepcopy(backbone)
        self.multiscale = multi_scale
        
        if self.multiscale:
            self.fpn = EffTrackFPN()
            
    def forward_template(self, z):
        if two_bbs:
            z = self.backbone2(z)[4]
            if extra_bb: # last layer of backbone also processes Template
                z = self.backbone2.left_stages[-1](z)
        else:    
            z = self.backbone(z)[4]
            if extra_bb: # last layer of backbone also processes Template
                z = self.backbone.left_stages[-1](z)
        return z

    def forward(self, x, z, template_anno=None):
        # 256 -> 16x16
        if self.multiscale:
            x = self.backbone(x, multi_scale=True)
            # print([i.shape for i in x])
            x = self.fpn(high=x[-1], mid=x[-2], low=x[-3])
        else:
            x = self.backbone(x)[4]
        
        if self.training or z.shape[1] == 3:
            if two_bbs:
                z = self.backbone2(z)[4]
                if extra_bb: # last layer of backbone also processes Template
                    z = self.backbone2.left_stages[-1](z)
            else:    
                z = self.backbone(z)[4]
                if extra_bb: # last layer of backbone also processes Template
                    z = self.backbone.left_stages[-1](z)
        
        return x, z

class EffTrackFPN(nn.Module):

    def __init__(self):
        super().__init__()
        
        temp = lambda in_ch, out_ch, kernel_size, stride: nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=1 if kernel_size==3 else 0),nn.BatchNorm2d(out_ch), nn.ReLU())
        
        self.low2mid = temp(40,80,3,2)
        
        self.midcomb = temp(160,160,1,1)

        self.high2mid = nn.Upsample(scale_factor=2, mode="nearest")
        
        self.final_comb = temp(320,160,1,1)
        
    # high: 160x16x16, mid: 80x32x32, low: 40x64x64
    def forward(self, high, mid, low):
        
        lowmid = self.low2mid(low) # 80x32x32
        midexp = self.midcomb(torch.cat([mid, lowmid], dim=1)) #160x32x32
        
        highmid = self.high2mid(high) # 160x32x32
        
        out = self.final_comb(torch.cat([highmid, midexp], dim=1))
        
        return out
    
    
class EffTrackNeck(nn.Module):

    def __init__(self, cfg=None, hidden_dim=160):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim

        self.query_att_down = True
        # LAYERS
        from lib.models.mobilevit_track.modules.lowtention import ConvAttention
        # self.fusion = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=hidden_dim//30, batch_first=True)
        self.fusion = ConvAttention(hidden_dim,head_dim_mul=1.0, querykey=True, convs=False, retAtt=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*4), nn.GELU(), nn.Linear(hidden_dim*4, hidden_dim))

        if self.query_att_down:
            self.query_down = ConvAttention(hidden_dim*(2 if extra_bb else 1), head_dim_mul=1.0, querykey=False, convs=False, retAtt=False, reduc_channel=0.5 if extra_bb else 1.0)
            if sep_queries:
                self.query_down_pos = ConvAttention(hidden_dim*(2 if extra_bb else 1), head_dim_mul=1.0, querykey=False, convs=False, retAtt=False, reduc_channel=0.5 if extra_bb else 1.0)
                self.fusion_pos = ConvAttention(hidden_dim,head_dim_mul=1.0, querykey=True, convs=False, retAtt=True)
             
    # x, z = self.neck(x, z)
    def forward(self, x, z):
        # x=N,160,16,16  z=N,160,8,8
        N,C,H,W = x.shape
        
        ## Add here to REDUCE it by SELF-ATTENTION
        if self.query_att_down:
            # query_embedding = z.mean(dim=(2,3)).reshape(N,1,C) # -> [N,1,C]
            query_in = z  #.reshape(N,C,-1)
            if sep_queries:
                query_embedding_pos = self.query_down_pos(query_in, mean_query=True).reshape(N,1,C)    
            query_embedding = self.query_down(query_in, mean_query=True).reshape(N,1,C)
        else:
            query_embedding = z.mean(dim=(2,3)).reshape(N,1,C) # -> [N,1,C]
            
        
        # x = x.reshape(N,C,-1).permute(0,2,1)
        ## Potential Feature enriching by Self-Attention
        
        
        # Association
        # feature_vec, attn_matrix = self.fusion(query=query_embedding, key=x, value=x, need_weights=True, average_attn_weights=True)
        if sep_queries:
            feature_vec_pos, _ = self.fusion_pos.forward_cross(query=query_embedding_pos.clone(), keyvalue=x)
        feature_vec, attn_matrix = self.fusion.forward_cross(query=query_embedding.clone(), keyvalue=x)
        # print("attout:",feature_vec.shape, "attn shape:",attn_matrix.shape )
        
        
        # Head average
        if detach_attn:
            attn_matrix = attn_matrix.detach().mean(dim=1)
        else:
            attn_matrix = attn_matrix.mean(dim=1)
            
        # mean=0,std=1 normalization
        attn_matrix = (attn_matrix - attn_matrix.mean(dim=(1,2), keepdim=True)) / attn_matrix.std(dim=(1,2), keepdim=True)
        
        # Transformer Finish
        if sep_queries:
            feature_vec_pos = feature_vec_pos + query_embedding
            
        query_embedding += feature_vec
        query_embedding = self.norm1(query_embedding)
        if True: # (Optional)
            query_embedding = self.mlp(query_embedding) + query_embedding
            query_embedding = self.norm2(query_embedding)
        
        if sep_queries:
            return (feature_vec_pos, query_embedding), attn_matrix 
        return query_embedding, attn_matrix


class EFFTrackHead(nn.Module):

    def __init__(self, cfg=None, hidden_dim=160):
        super().__init__()
        self.cfg = cfg
        self.fmap_size = self.cfg.DATA.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        if multi_scale:
            self.fmap_size *= 2 
        hidfac = 4
        
        if sep_queries:
            self.mlp_score = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim*hidfac),
                        nn.ReLU(),
                        nn.Linear(hidden_dim*hidfac, hidden_dim*hidfac),
                        nn.ReLU(),
                        nn.Linear(hidden_dim*hidfac, hidden_dim*hidfac),
                        nn.ReLU(),
                        nn.Linear(hidden_dim*hidfac,1), # posoffset+sizeoffset
            )
            self.mlp_pos = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim*hidfac),
                        nn.ReLU(),
                        nn.Linear(hidden_dim*hidfac, hidden_dim*hidfac),
                        nn.ReLU(),
                        nn.Linear(hidden_dim*hidfac, hidden_dim*hidfac),
                        nn.ReLU(),
                        nn.Linear(hidden_dim*hidfac,4), # posoffset+sizeoffset
            )
        else:
            self.mlp = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim*hidfac),
                        nn.ReLU(),
                        nn.Linear(hidden_dim*hidfac, hidden_dim*hidfac),
                        nn.ReLU(),
                        nn.Linear(hidden_dim*hidfac, hidden_dim*hidfac),
                        nn.ReLU(),
                        nn.Linear(hidden_dim*hidfac,5), # score+posoffset+sizeoffset
            )
        
        self.blur = torchvision.transforms.GaussianBlur(3, sigma=1.0)
        
        self.sigmoid = torch.nn.Sigmoid()
    
    def print_attmap(self, attnmap):
        # return
        prstr = ""
        for i in range(attnmap.shape[1]):
            for k in range(attnmap.shape[2]):
                prstr += str(attnmap[0,i,k].item())[:5] + ","
            prstr += "\n"
        print(prstr)
    
    # TODO DETACH AT SOME POINT!!!!! Like Attnmap!
    # [N,16,16]
    def process_attnmap(self, attnmap):
        spat_size = attnmap.shape[-1]
        # Init
        N = attnmap.shape[0]
        prepos = torch.zeros(N,2).cuda()
        presize = torch.zeros(N,2).cuda()
        
        # Before blur
        # self.print_attmap(attnmap)
        # print(torch.argmax(attnmap.reshape(N,-1), dim=1)[0]% self.fmap_size, torch.argmax(attnmap.reshape(N,-1), dim=1)[0]// self.fmap_size, )
        
        ## BLUR
        attnmap_blurred = self.blur(attnmap.unsqueeze(1)).squeeze(1)
        attnmap_blurred = (attnmap_blurred - attnmap_blurred.mean(dim=(1,2), keepdim=True)) / (attnmap_blurred.std(dim=(1,2), keepdim=True))
        
        # After blur
        # self.print_attmap(attnmap)
        # print(torch.argmax(attnmap.reshape(N,-1), dim=1)[0]% self.fmap_size, torch.argmax(attnmap.reshape(N,-1), dim=1)[0]// self.fmap_size, )
        
        
        ## FIND Prepos
        indices = torch.argmax(attnmap_blurred.reshape(N,-1), dim=1) # -> [N]
        
        ## Prepos index conversion
        # x
        prepos[:,0] = indices % self.fmap_size
        # y
        prepos[:,1] = indices // self.fmap_size
        prepos = prepos.int()
        
        
        window = (spat_size // 8) - 1
        # print(spat_size, window, attnmap.shape)
        ## Presize
        for i in range(N):
            # H
            temp = self.sigmoid(attnmap[i,:,max(0,prepos[i,0]-window):min(self.fmap_size,prepos[i,0]+window)]).clamp(min=1e-4, max=1 - 1e-4).mean()
            presize[i,0] += temp 
            # W 
            temp2 =   self.sigmoid(attnmap[i,max(0,prepos[i,1]-window):min(self.fmap_size, prepos[i,1]+window),:]).clamp(min=1e-4, max=1 - 1e-4).mean()
            presize[i,1] += temp2
        # print(presize[0,:])
        
        return prepos, presize * self.fmap_size
    
    # query: [N,memquery,160], attnmatrix: [N,memquery,16*16]
    def forward(self, query_embedding, attn_matrix):
        if sep_queries:
            feature_pos, query_embedding = query_embedding 
        N = query_embedding.shape[0]
        
        ## Mean memory queries dimension
        query_embedding = query_embedding.mean(dim=1) # -> [N,160]
        if sep_queries:
            feature_pos = feature_pos.mean(dim=1) 
        attn_matrix = attn_matrix.mean(dim=1).reshape(N, self.fmap_size, self.fmap_size) # -> [N,16,16]
        
        
        ## Query processing HEAD
        if sep_queries:
            score = self.mlp_score(query_embedding).reshape(N)
            bbox_info = self.mlp_pos(feature_pos)
            sizeoffset, posoffset = bbox_info[:,:2], bbox_info[:,2:]
        else:
            bbox_info = self.mlp(query_embedding)
            # bbox_info[:,:1] = bbox_info[:,:1].sigmoid_().clamp(min=1e-4, max=1 - 1e-4) 
            score, sizeoffset, posoffset = bbox_info[:,0], bbox_info[:,1:3], bbox_info[:,3:5]
        
        ## Attention Map processing
        prepos, presize = self.process_attnmap(attnmap=attn_matrix)

        if clamp_offset:
            sizeoffset, posoffset = sizeoffset.clamp(min=-2, max=2), posoffset.clamp(min=-2, max=2 )
        
        # print("offsets size/pos:",sizeoffset[:2,:], posoffset[:2,:])
        # only score uses sigmoid, because we have posoffset and sizeoffset!
        # TODO: Tryout with sigmoid everywhere!! 
        bbox = torch.cat([(prepos + posoffset)/self.fmap_size, (presize+sizeoffset)/self.fmap_size],dim=1)
        
        if predict_center: # assume predicted position is the center!!!
            bbox[:,:2] = bbox[:,:2] - 0.5 * bbox[:,2:]
        ## IMPORTANT: Loss gets [0,1] pos/size
        return bbox, score


def build_mobilevitv2_track(cfg, settings=None, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if "mobilevitv2" in cfg.MODEL.BACKBONE.TYPE:
        width_multiplier = float(cfg.MODEL.BACKBONE.TYPE.split('-')[-1])
        backbone = create_mobilevitv2_backbone(pretrained, width_multiplier, has_mixed_attn=cfg.MODEL.BACKBONE.MIXED_ATTN, training=training)
        if cfg.MODEL.BACKBONE.MIXED_ATTN is True:
            backbone.mixed_attn = True
        else:
            backbone.mixed_attn = False
        hidden_dim = backbone.model_conf_dict['layer4']['out']
        patch_start_index = 1
    elif "lowformer" in cfg.MODEL.BACKBONE.TYPE:
        from lib.models.mobilevit_track.lowformer_track import build_lowformer_backbone
        backbone = build_lowformer_backbone(cfg.MODEL.BACKBONE.TYPE.replace("lowformer_",""), cfg)
        hidden_dim = cfg.MODEL.HEAD.NUM_CHANNELS
        if cfg.MODEL.BACKBONE.EFFTRACK:
            print("Efficient Tracking Model initializing...")
            backbone = EffTrackWrapper(backbone, training=training)
            x,z = backbone(torch.randn(2,3,256,256),torch.randn(2,3,128,128))
            print("backbone output x/z:",x.shape, z.shape)
        else:
            backbone = LowFormerWrapper(backbone, training=training, type_emb=cfg.MODEL.TYPE_EMB, type_embv2=cfg.MODEL.TYPE_EMBV2)
    else:
        raise NotImplementedError

    # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    # build neck module to fuse template and search region features
    if cfg.MODEL.NECK:
        if cfg.MODEL.NECK.TYPE == "EFFTRACK":
            print("Initializing EffTrack neck...")
            neck = EffTrackNeck(cfg=cfg, hidden_dim=hidden_dim).cuda()
            if training:
                query_embed, attn = neck(x.cuda(), z.cuda())
                try:
                    print("neck output q/att:",query_embed.shape, attn.shape, attn.sum(dim=-1))
                except:
                    print("neck output q/att:",query_embed[0].shape, query_embed[1].shape, attn.shape, attn.sum(dim=-1))
                
        else:
            neck = build_neck(cfg=cfg, hidden_dim=hidden_dim)
    else:
        neck = nn.Identity()

    if cfg.MODEL.NECK.TYPE == "BN_PWXCORR":
        try:
            in_features = backbone.model_conf_dict['layer4']['out']
        except:
            in_features = hidden_dim
        feature_fusor = build_feature_fusor(cfg=cfg, in_features = in_features,
                                               xcorr_out_features=cfg.MODEL.NECK.NUM_CHANNS_POST_XCORR)
    elif cfg.MODEL.NECK.TYPE == "BN_SSAT" or cfg.MODEL.NECK.TYPE == "BN_HSSAT":
        feature_fusor = build_feature_fusor(cfg=cfg, in_features = backbone.model_conf_dict['layer4']['out'],
                                               xcorr_out_features=None)
    elif cfg.MODEL.HEAD.TYPE == "EFFTRACK":
        feature_fusor = None
    else:
        raise NotImplementedError

    # build head module
    if cfg.MODEL.HEAD.TYPE == "EFFTRACK":
        box_head = EFFTrackHead(cfg, hidden_dim=hidden_dim).cuda()
        if training:
            with torch.no_grad():
                out = box_head(*(query_embed, attn ))
                print(out)
        
    else:
        box_head = build_box_head(cfg, cfg.MODEL.HEAD.NUM_CHANNELS)

    ## REMOVE TODO
    torch.autograd.set_detect_anomaly(True)
    
    model = MobileViTv2_Track(
        backbone=backbone,
        neck=neck,
        feature_fusor=feature_fusor,
        box_head=box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'mobilevit_track' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)

        assert missing_keys == [] and unexpected_keys == [], "The backbone layers do not exactly match with the " \
                                                             "checkpoint state dictionaries. Please have a look at " \
                                                             "what those missing keys are!"

        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
    
    # print("cfg:",cfg)
    # print("settings:",str(settings), vars(settings))
    # settings.save_dir
    # ckfolder = os.path.join("output/checkpoints/",settings.project_path)
    # cks = os.listdir(ckfolder)
    # if len(cks)>0 and False:
    #     ck_chosen = sorted(cks, key=lambda x: int(x.split("ep")[-1].replace(".pth.tar","")))[-1]
    #     ckpath = os.path.join(ckfolder,ck_chosen)
    #     checkpoint = torch.load(ckpath, map_location="cpu")
    #     missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
    #     print("Checkpoint loaded from:", ckpath)
    # from tracking.myutils import to_file
    # to_file(str(model),"modelprint.txt")
    
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