"""
Basic MobileViT-Track model.
"""
import math
import os
from typing import List

import torch, torchvision
from torch import nn
from torch.nn.modules.transformer import _get_clones
from torchvision.ops import FeaturePyramidNetwork

from lib.models.mobilevit_track.layers.conv_layer import Conv2d
from .layers.neck import build_neck, build_feature_fusor
from .layers.head import build_box_head

from lib.models.mobilevit_track.mobilevit_v2 import MobileViTv2_backbone
from lib.utils.box_ops import box_xyxy_to_cxcywh
from easydict import EasyDict as edict
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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

        #print("after bb:", x.shape, z.shape)
        # Forward neck
        x, z = self.neck(x, z)

        #print("after neck:", x.shape, z.shape)
        # Forward feature fusor
        if self.feature_fusor is not None:
            feat_fused = self.feature_fusor(z, x)
        else:
            feat_fused = (x,z) # actually query+attnmap

        #print("after ff:", x.shape, z.shape)
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
            max_score = 0
            # run the center head
            if self.head_type == "CENTER_SSAT_LOWFORM":
                score_map_ctr, bbox, size_map, offset_map, max_score = self.box_head(opt_feat, gt_score_map)
            else:
                score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,
                   'max_score': max_score}
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



# extra_bb = True 
multicross = True
# multi_scale = True
# two_bbs = True
# sep_queries = False
clamp_offset = True
detach_attn = False ##
predict_center = True
multicross = True

class EffTrackWrapper(nn.Module):

    def __init__(self, cfg, backbone, training, hidden_dim):
        super().__init__()
        self.cfg = cfg
        two_bbs = self.cfg.TWO_BBS
        extra_bb = self.cfg.EXTRA_BB
        
        self.backbone = backbone
        self.training = training
        out_channels = hidden_dim
        out_channels_template = hidden_dim * 2

        # Define channel list for the SEARCH ('x') path --> NO extra_bb logic
        in_channels_list_search = [hidden_dim // 4, hidden_dim // 2, hidden_dim]

        # Define channel list for the TEMPLATE ('z') path --> Accounts for extra_bb logic
        in_channels_list_template = [
            hidden_dim // 4, 
            hidden_dim // 2, 
            hidden_dim * 2 if extra_bb else hidden_dim
        ]

        # Create the FPN for the search path with its specific channel list
        self.fpn = FeaturePyramidNetwork(in_channels_list_search, out_channels)
        
        if two_bbs:
            import copy
            self.backbone2 = copy.deepcopy(backbone)
            # Create the FPN for the template path with its specific channel list
            self.fpn2 = FeaturePyramidNetwork(in_channels_list_template, out_channels_template)
            
    def forward_template(self, z):
        two_bbs = self.cfg.TWO_BBS
        extra_bb = self.cfg.EXTRA_BB

        # Determine which backbone and FPN to use
        backbone_z = self.backbone2 if two_bbs else self.backbone
        fpn_z = self.fpn2 if two_bbs else self.fpn

        # Get features from the backbone
        z_features = backbone_z(z)
        z_last_three_list = [z_features[2], z_features[3], z_features[4]]  # layers 2, 3, 4
        
        # Apply the extra backbone stage if needed (before FPN)
        if extra_bb:
            z_bb_extra = self.backbone2 if two_bbs else self.backbone
            z_last_three_list[-1] = z_bb_extra.left_stages[-1](z_last_three_list[-1])
            
        # Prepare features for FPN (dictionary format)
        z_last_three_dict = {str(i): f for i, f in enumerate(z_last_three_list)}

        # Process features with FPN
        z_fpn_out = fpn_z(z_last_three_dict)
        
        # Return a list of feature tensors
        return list(z_fpn_out.values())

    def forward(self, x, z, template_anno=None):
        # This code path is for multicross=True
        # --- Process Search Region 'x' ---
        # The search image 'x' is always a tensor and needs to be processed.
        x_features = self.backbone(x)
        x_last_three_list = [x_features[2], x_features[3], x_features[4]]
        x_last_three_dict = {str(i): f for i, f in enumerate(x_last_three_list)}
        x_fpn_out = self.fpn(x_last_three_dict)
        x_fpn_out_list = list(x_fpn_out.values())

        # --- Process Template 'z' ---
        if isinstance(z, torch.Tensor):
            # This block runs during TRAINING or on the FIRST frame of inference.
            # 'z' is an image tensor, so we process it with backbone and FPN.
            z_last_three = self.forward_template(z)
        else:
            # This block runs during subsequent frames of INFERENCE.
            # 'z' is already a list of FPN-processed feature tensors.
            z_last_three = z
                                    
        return x_fpn_out_list, z_last_three
    

# The old EffTrackFPN is no longer needed and has been removed.

    
class EffTrackNeck(nn.Module):

    def __init__(self, cfg=None, hidden_dim=160):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.query_att_down = True
        
        # LAYERS - Create 3 cross-attention modules for the 3 layer pairs
        from lib.models.mobilevit_track.modules.lowtention import ConvAttention
        
        
        # Three fusion modules for the three layer pairs. All operate on hidden_dim now.
        self.fusion_layer2 = ConvAttention(hidden_dim, head_dim_mul=1.0, querykey=True, convs=False, retAtt=True)
        self.fusion_layer3 = ConvAttention(hidden_dim, head_dim_mul=1.0, querykey=True, convs=False, retAtt=True)
        self.fusion_layer4 = ConvAttention(hidden_dim, head_dim_mul=1.0, querykey=True, convs=False, retAtt=True)
        
        # Projection layers are removed as FPN handles uniform channel dimension.
        
        # MLP and Normalization layers for each of the 3 streams
        mlp_block = lambda: nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 12),
            nn.GELU(),
            nn.Linear(hidden_dim * 12, hidden_dim)
        )
        
        self.norm1_l2, self.norm2_l2 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.mlp_l2 = mlp_block()

        self.norm1_l3, self.norm2_l3 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.mlp_l3 = mlp_block()

        self.norm1_l4, self.norm2_l4 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.mlp_l4 = mlp_block()
        
        sep_queries = self.cfg.SEP_QUERIES
        extra_bb = self.cfg.EXTRA_BB
        
        if self.query_att_down:
            # Adjust for concatenated features
            self.query_down = self.query_down = ConvAttention(hidden_dim*(2 if extra_bb else 1), head_dim_mul=1.0, querykey=False, convs=False, retAtt=False, reduc_channel=0.5 if extra_bb else 1.0)
            encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
            self.query_encoder = TransformerEncoder(encoder_layer, num_layers=1)
            self.query_pos_embed = nn.Parameter(torch.zeros(1, 3, hidden_dim))
            nn.init.trunc_normal_(self.query_pos_embed, std=.02)

            # New encoder for the final aggregated feature vectors
            output_encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
            self.output_encoder = TransformerEncoder(output_encoder_layer, num_layers=1)
            self.output_pos_embed = nn.Parameter(torch.zeros(1, 3, hidden_dim))
            nn.init.trunc_normal_(self.output_pos_embed, std=.02)
            if sep_queries:
                self.query_down_pos = ConvAttention(
                    hidden_dim, 
                    head_dim_mul=1.0, 
                    querykey=False, 
                    convs=False, 
                    retAtt=False, 
                    reduc_channel=0.5 if extra_bb else 1.0
                )
                self.fusion_pos_layer2 = ConvAttention(hidden_dim, head_dim_mul=1.0, querykey=True, convs=False, retAtt=True)
                self.fusion_pos_layer3 = ConvAttention(hidden_dim, head_dim_mul=1.0, querykey=True, convs=False, retAtt=True)
                self.fusion_pos_layer4 = ConvAttention(hidden_dim, head_dim_mul=1.0, querykey=True, convs=False, retAtt=True)

    def forward(self, x, z):
        """
        x: search region features (list of 3 tensors)
        z: template features (list of 3 tensors)
        """
        if multicross:
            return self.forward_multilayer(x, z)
        else:
            return self.forward_original(x, z)
    
    def forward_multilayer(self, x_last_three, z_last_three):
        """
        Forward pass with multi-layer cross attention. Queries for all three template
        layers are prepared and (optionally) self-encoded together BEFORE any
        cross-attention, so every fusion_layer uses the same encoded queries.
        """
        # Get batch size and feature dimension from the last feature map
        N, C, W, H = x_last_three[-1].shape

        # Container for attention maps (per layer)
        attn_matrices = []

        # --- Build initial queries for all three template layers ---
        z2_in = z_last_three[0]
        z3_in = z_last_three[1]
        z4_in = z_last_three[2]

        if self.query_att_down:
            z2_query = self.query_down(z2_in, mean_query=True).reshape(N, 1, C)
            z3_query = self.query_down(z3_in, mean_query=True).reshape(N, 1, C)
            z4_query = self.query_down(z4_in, mean_query=True).reshape(N, 1, C)

            # Encode all three queries together so the encoder can share context
            queries = torch.cat([z2_query, z3_query, z4_query], dim=1)  # [N, 3, C]
            queries = queries + self.query_pos_embed
            encoded_queries = self.query_encoder(queries)  # [N, 3, C]
            z2_query, z3_query, z4_query = encoded_queries[:, 0:1, :], encoded_queries[:, 1:2, :], encoded_queries[:, 2:3, :]
        else:
            z2_query = z2_in.mean(dim=(2, 3)).reshape(N, 1, C)
            z3_query = z3_in.mean(dim=(2, 3)).reshape(N, 1, C)
            z4_query = z4_in.mean(dim=(2, 3)).reshape(N, 1, C)

        # --- Layer 2 Cross-Attention + FFN ---
        feat_vec_2, attn_2 = self.fusion_layer2.forward_cross(query=z2_query, keyvalue=x_last_three[0])
        out_vec_2 = z2_query + feat_vec_2
        out_vec_2 = self.norm1_l2(out_vec_2)
        out_vec_2 = self.mlp_l2(out_vec_2) + out_vec_2
        out_vec_2 = self.norm2_l2(out_vec_2)
        attn_matrices.append(attn_2)

        # --- Layer 3 Cross-Attention + FFN ---
        feat_vec_3, attn_3 = self.fusion_layer3.forward_cross(query=z3_query, keyvalue=x_last_three[1])
        out_vec_3 = z3_query + feat_vec_3
        out_vec_3 = self.norm1_l3(out_vec_3)
        out_vec_3 = self.mlp_l3(out_vec_3) + out_vec_3
        out_vec_3 = self.norm2_l3(out_vec_3)
        attn_matrices.append(attn_3)

        # --- Layer 4 Cross-Attention + FFN ---
        feat_vec_4, attn_4 = self.fusion_layer4.forward_cross(query=z4_query, keyvalue=x_last_three[2])
        out_vec_4 = z4_query + feat_vec_4
        out_vec_4 = self.norm1_l4(out_vec_4)
        out_vec_4 = self.mlp_l4(out_vec_4) + out_vec_4
        out_vec_4 = self.norm2_l4(out_vec_4)
        attn_matrices.append(attn_4)

        # --- Final Aggregation and Encoding ---
        # Keep the same ordering as before (deepest first)
        output_vectors = torch.cat([out_vec_4, out_vec_3, out_vec_2], dim=1)  # [N, 3, C]
        # output_vectors = output_vectors + self.output_pos_embed
        # encoded_outputs = self.output_encoder(output_vectors)  # [N, 3, C]

        # Flatten for the head: [N, 1, C*3]
        concatenated_features = output_vectors.reshape(N, 1, -1)

        # Return concatenated feature and (deepest) attention map for compatibility
        final_attn_matrix = attn_4

        return concatenated_features, final_attn_matrix

    # x, z = self.neck(x, z)
    def forward_original(self, x, z):
        # x=N,160,16,16  z=N,160,8,8
        N,C,H,W = x.shape

        sep_queries = self.cfg.SEP_QUERIES

        ## Add here to REDUCE it by SELF-ATTENTION
        ## TODO Francesco: [B,7,7,256] -> [B,1,256] 
        if self.query_att_down:
            query_embedding = z.mean(dim=(2,3)).reshape(N,1,2*C) # -> [N,1,C]
            query_in = z  #.reshape(N,C,-1)
            if sep_queries:
                query_embedding_pos = self.query_down_pos(query_in, mean_query=True).reshape(N,1,C)    
            query_embedding = self.query_down(query_in, mean_query=True).reshape(N,1,C)
        else:
            query_embedding = z.mean(dim=(2,3)).reshape(N,1,C) # -> [N,1,C]
            
        
        # x = x.reshape(N,C,-1).permute(0,2,1)
        ## Potential Feature enriching by Self-Attention
        
        
        # Association TODO FRANCESCO
        # feature_vec, attn_matrix = self.fusion(query=query_embedding, key=x, value=x, need_weights=True, average_attn_weights=True)
        if sep_queries:
            feature_vec_pos, _ = self.fusion_pos.forward_cross(query=query_embedding_pos.clone(), keyvalue=x)
        feature_vec, attn_matrix = self.fusion.forward_cross(query=query_embedding.clone(), keyvalue=x)
        # print("attout:",feature_vec.shape, "attn shape:",attn_matrix.shape )
        
        

        # Transformer Finish
        if sep_queries:
            feature_vec_pos = feature_vec_pos + query_embedding
            
        query_embedding += feature_vec
        query_embedding = self.norm1(query_embedding)
        if True: # (Optional)
            query_embedding = self.mlp(query_embedding) + query_embedding
            query_embedding = self.norm2(query_embedding)
        
        if sep_queries:
            return (feature_vec_pos, query_embedding) 
        return feature_vec, query_embedding



## Gets Query Embedding and AttentionMAP
class EFFTrackHead(nn.Module):
    def __init__(self, cfg=None, hidden_dim=160):
        super().__init__()
        self.cfg = cfg
        self.fmap_size = self.cfg.DATA.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        multi_scale = self.cfg.MULTI_SCALE

        if multi_scale:
            self.fmap_size *= 2
        hidfac = 4
        sep_queries = self.cfg.SEP_QUERIES

        # The input dimension for the head's MLPs is now 3x the base hidden_dim
        input_dim = hidden_dim * 3 if multicross else hidden_dim

        if sep_queries:
            self.mlp_score = nn.Sequential(
                # Corrected: Use input_dim instead of hidden_dim
                nn.Linear(input_dim, hidden_dim * hidfac),
                nn.ReLU(),
                nn.Linear(hidden_dim * hidfac, hidden_dim * hidfac),
                nn.ReLU(),
                nn.Linear(hidden_dim * hidfac, 1),
            )
            self.mlp_pos = nn.Sequential(
                # Corrected: Use input_dim instead of hidden_dim
                nn.Linear(input_dim, hidden_dim * hidfac),
                nn.ReLU(),
                nn.Linear(hidden_dim * hidfac, hidden_dim * hidfac),
                nn.ReLU(),
                nn.Linear(hidden_dim * hidfac, 4),  # posoffset+sizeoffset
            )
        else:
            self.mlp = nn.Sequential(
                # Corrected: Use input_dim instead of hidden_dim
                nn.Linear(input_dim, hidden_dim * hidfac),
                nn.ReLU(),
                nn.Linear(hidden_dim * hidfac, hidden_dim * hidfac),
                nn.ReLU(),
                nn.Linear(hidden_dim * hidfac, 5),  # score+posoffset+sizeoffset
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
    # def process_attnmap(self, attnmap):
    #     spat_size = attnmap.shape[-1]
    #     # Init
    #     N = attnmap.shape[0]
    #     prepos = torch.zeros(N,2).cuda()
    #     presize = torch.zeros(N,2).cuda()
        
    #     # Before blur
    #     # self.print_attmap(attnmap)
    #     # print(torch.argmax(attnmap.reshape(N,-1), dim=1)[0]% self.fmap_size, torch.argmax(attnmap.reshape(N,-1), dim=1)[0]// self.fmap_size, )
        
    #     ## BLUR
    #     attnmap_blurred = self.blur(attnmap.unsqueeze(1)).squeeze(1)
    #     attnmap_blurred = (attnmap_blurred - attnmap_blurred.mean(dim=(1,2), keepdim=True)) / (attnmap_blurred.std(dim=(1,2), keepdim=True))
        
    #     # After blur
    #     # self.print_attmap(attnmap)
    #     # print(torch.argmax(attnmap.reshape(N,-1), dim=1)[0]% self.fmap_size, torch.argmax(attnmap.reshape(N,-1), dim=1)[0]// self.fmap_size, )
        
        
    #     ## FIND Prepos
    #     indices = torch.argmax(attnmap_blurred.reshape(N,-1), dim=1) # -> [N]
        
    #     ## Prepos index conversion
    #     # x
    #     prepos[:,0] = indices % self.fmap_size
    #     # y
    #     prepos[:,1] = indices // self.fmap_size
    #     prepos = prepos.int()
        
        
    #     window = (spat_size // 8) - 1
    #     # print(spat_size, window, attnmap.shape)
    #     ## Presize
    #     for i in range(N):
    #         # H
    #         temp = self.sigmoid(attnmap[i,:,max(0,prepos[i,0]-window):min(self.fmap_size,prepos[i,0]+window)]).clamp(min=1e-4, max=1 - 1e-4).mean()
    #         presize[i,0] += temp 
    #         # W 
    #         temp2 =   self.sigmoid(attnmap[i,max(0,prepos[i,1]-window):min(self.fmap_size, prepos[i,1]+window),:]).clamp(min=1e-4, max=1 - 1e-4).mean()
    #         presize[i,1] += temp2
    #     # print(presize[0,:])
        
    #     return prepos, presize * self.fmap_size
    
    # query: [N,memquery,160], attnmatrix: [N,memquery,16*16]
    def forward(self, query_embedding, attn_matrix):
        sep_queries = self.cfg.SEP_QUERIES

        if sep_queries:
            feature_pos, query_embedding = query_embedding
        
        # Squeeze the middle dimension (from [N, 1, C]) to get [N, C]
        query_embedding = query_embedding.squeeze(1)

        if sep_queries:
            feature_pos = feature_pos.squeeze(1)
            score = self.mlp_score(query_embedding)  # Output: [N, 1]
            bbox_info = self.mlp_pos(feature_pos)   # Output: [N, 4]
            # No need for separate size/pos, mlp_pos directly predicts the 4 values
            bbox = bbox_info
            
        else:
            # mlp input is the concatenated feature vector
            bbox_info = self.mlp(query_embedding) # Output: [N, 5]
            
            # Unpack the 5 outputs: 1 for score, 2 for pos, 2 for size
            score = bbox_info[:, 0:1] # Keep dimension: [N, 1]
            pos = bbox_info[:, 1:3]
            size = bbox_info[:, 3:5]
            bbox = torch.cat([pos, size], dim=1) # [xywh]

        # The output from the head should have a consistent shape
        # The loss function likely expects shape [N, 1, 4] for boxes and [N, 1] for scores
        return bbox.unsqueeze(1), score


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
            # Pass hidden_dim to the wrapper
            backbone = EffTrackWrapper(cfg, backbone, training=training, hidden_dim=hidden_dim)
            x,z = backbone(torch.randn(2,3,256,256),torch.randn(2,3,128,128))
            #print("backbone output x/z:",x.shape, z.shape)
        else:
            backbone = LowFormerWrapper(backbone, training=training, type_emb=cfg.MODEL.TYPE_EMB, type_embv2=cfg.MODEL.TYPE_EMBV2)
    else:
        raise NotImplementedError

    # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    if False:
        print("Test backbone:",backbone.backbone(torch.randn(1,3,224,224))[4].shape)
        print(backbone.backbone)
    # build neck module to fuse template and search region features
    if cfg.MODEL.NECK:
        if cfg.MODEL.NECK.TYPE == "EFFTRACK":
            print("Initializing EffTrack neck...")
            neck = EffTrackNeck(cfg=cfg, hidden_dim=hidden_dim).cuda()
            if training and multicross:
                x_cuda = [t.cuda() for t in x]
                z_cuda = [t.cuda() for t in z]
                
                # Pass the lists of cuda tensors to the neck
                query_embed, attn = neck(x=x_cuda, z=z_cuda)
                
                # You might need to adjust the print statement based on the neck's new output
                try:
                    # For cases with separate pos/score queries
                    print("neck output pos_q/score_q:", query_embed[0].shape, query_embed[1].shape)
                except:
                    # For single query output
                    print("neck output q:", query_embed.shape)
                print("neck output attn:", attn.shape)
            elif training:
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
    # torch.autograd.set_detect_anomaly(True)
    
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