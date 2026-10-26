"""
Basic MobileViT-Track model.
"""
import copy
import os

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.transformer import _get_clones
from torchvision.ops import FeaturePyramidNetwork

from .layers.neck import build_neck, build_feature_fusor
from .layers.head import build_box_head
from lib.models.mobilevit_track.mobilevit_v2 import MobileViTv2_backbone
from lib.utils.box_ops import box_xyxy_to_cxcywh


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
        if template_anno is None:
            x, z = self.backbone(x=search, z=template)
        else:
            x, z = self.backbone(x=search, z=template, template_anno=template_anno)

        x, z = self.neck(x, z)

        if self.feature_fusor is not None:
            feat_fused = self.feature_fusor(z, x)
        else:
            feat_fused = (x, z)  # actually query + attention map

        return self.forward_head(feat_fused, None)

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

        if self.head_type == "EFFTRACK":
            bbox, scores = self.box_head(*backbone_feature)
            return {'pred_boxes': bbox.view(bs, 1, 4),
                    'scores': scores,
                    }
        elif self.head_type == "CORNER":
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            return {'pred_boxes': outputs_coord_new,
                    'score_map': score_map,
                    }
        elif "CENTER" in self.head_type:
            max_score = 0
            if self.head_type == "CENTER_SSAT_LOWFORM":
                score_map_ctr, bbox, size_map, offset_map, max_score = self.box_head(opt_feat, gt_score_map)
            else:
                score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord_new = bbox.view(bs, 1, 4)
            return {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map,
                    'max_score': max_score}
        else:
            raise NotImplementedError


class LowFormerWrapper(nn.Module):
    """Runs the LowFormer backbone stage-by-stage on search ('x') and template ('z'),
    merging the two into a single batch for the last stage so it can attend across them.
    """

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
            if type_embv2:
                torch.nn.init.trunc_normal_(self.token_type_embed, mean=1.0, std=0.02, a=-1.0, b=3.0)
            else:
                torch.nn.init.trunc_normal_(self.token_type_embed, std=0.02, a=-2.0, b=2.0)

    def forward(self, x, z, template_anno=None):
        orig_inp_size = int(x.shape[2] // 2)
        x = self.backbone.input_stem(x)
        if self.training_it:
            z = self.backbone.input_stem(z)

        last_stage = len(self.backbone.stages) - 1
        for ind, stage in enumerate(self.backbone.stages):
            if ind != last_stage:
                x = stage(x)
                if self.training_it or ind > 1:
                    z = stage(z)
                continue

            # Last stage: optionally add a token-type embedding that tells the
            # network apart which tokens come from the search region vs. the
            # template (and, within the template, which tokens fall on the target).
            if template_anno is not None and self.type_emb:
                # template_anno is in template space, [1, N, 4] (x, y, w, h)
                type_mask = torch.zeros(z.shape[0], 1, z.shape[2], z.shape[3])
                for i in range(x.shape[0]):
                    xcord, ycord, w, h = (int(template_anno[0, i, 0] * orig_inp_size / 8),
                                          int(template_anno[0, i, 1] * orig_inp_size / 8),
                                          int(template_anno[0, i, 2] * orig_inp_size / 8),
                                          int(template_anno[0, i, 3] * orig_inp_size / 8))
                    xcord, ycord = max(0, xcord), max(ycord, 0)
                    w, h = min(xcord + w, 14) - xcord, min(ycord + h, 14) - ycord
                    type_mask[i, :, ycord:ycord + h, xcord:xcord + w] = 1
                type_mask = type_mask.cuda()
            elif self.type_emb:
                type_mask = torch.zeros(z.shape[0], 1, z.shape[2], z.shape[3]).cuda()
                h, w = z.shape[2], z.shape[3]
                type_mask[:, :, int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)] = 1

            if self.type_embv2:
                x = x * self.token_type_embed[0, :].reshape(1, z.shape[1], 1, 1)
                z = z * (type_mask * self.token_type_embed[1, :].reshape(1, z.shape[1], 1, 1)
                         + (1 - type_mask) * self.token_type_embed[2, :].reshape(1, z.shape[1], 1, 1))
            elif self.type_emb:
                x = x + self.token_type_embed[0, :].reshape(1, z.shape[1], 1, 1)
                z = z + (type_mask * self.token_type_embed[1, :].reshape(1, z.shape[1], 1, 1)
                         + (1 - type_mask) * self.token_type_embed[2, :].reshape(1, z.shape[1], 1, 1))

            merged = torch.cat([x, torch.cat([z, z.clone()], dim=3)], dim=2)
            merged = stage(merged)
            x, z = merged[:, :, :z.shape[2], :], merged[:, :, z.shape[2]:, :int(z.shape[3] // 2)]

        return x, z


class EffTrackWrapper(nn.Module):
    """Runs the (LowFormer) backbone on search and template separately, each
    followed by a Feature Pyramid Network so the neck gets 3 same-channel
    feature maps per stream. Optionally uses a second, independently-weighted
    backbone/FPN for the template (TWO_BBS)."""

    def __init__(self, cfg, backbone, training, hidden_dim):
        super().__init__()
        self.cfg = cfg
        self.training = training
        self.two_bbs = cfg.TWO_BBS
        self.extra_bb = cfg.EXTRA_BB

        self.backbone = backbone
        out_channels = hidden_dim
        out_channels_template = hidden_dim * 2

        in_channels_list_search = [hidden_dim // 4, hidden_dim // 2, hidden_dim]
        in_channels_list_template = [
            hidden_dim // 4,
            hidden_dim // 2,
            hidden_dim * 2 if self.extra_bb else hidden_dim,
        ]

        self.fpn = FeaturePyramidNetwork(in_channels_list_search, out_channels)

        if self.two_bbs:
            self.backbone2 = copy.deepcopy(backbone)
            self.fpn2 = FeaturePyramidNetwork(in_channels_list_template, out_channels_template)

    def forward_template(self, z):
        backbone_z = self.backbone2 if self.two_bbs else self.backbone
        fpn_z = self.fpn2 if self.two_bbs else self.fpn

        z_features = backbone_z(z)
        z_last_three = [z_features[2], z_features[3], z_features[4]]  # layers 2, 3, 4

        if self.extra_bb:
            z_last_three[-1] = backbone_z.left_stages[-1](z_last_three[-1])

        z_fpn_out = fpn_z({str(i): f for i, f in enumerate(z_last_three)})
        return list(z_fpn_out.values())

    def forward(self, x, z, template_anno=None):
        # --- Search region 'x': always an image tensor. ---
        x_features = self.backbone(x)
        x_last_three = [x_features[2], x_features[3], x_features[4]]
        x_fpn_out = self.fpn({str(i): f for i, f in enumerate(x_last_three)})
        x_fpn_out_list = list(x_fpn_out.values())

        # --- Template 'z': an image tensor during training / the first inference
        # frame, or an already FPN-processed feature list on later inference frames. ---
        if isinstance(z, torch.Tensor):
            z_last_three = self.forward_template(z)
        else:
            z_last_three = z

        return x_fpn_out_list, z_last_three


class EffTrackNeck(nn.Module):
    """Cross-attends each of the 3 template feature-pyramid levels (as query)
    against the corresponding search-region level (as key/value), producing one
    fused vector per level that the head then reads out box + score from."""

    def __init__(self, cfg=None, hidden_dim=160):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim

        from lib.models.mobilevit_track.modules.lowtention import ConvAttention

        # One cross-attention fusion module per pyramid level.
        self.fusion_layer2 = ConvAttention(hidden_dim, head_dim_mul=1.0, querykey=True, convs=False, retAtt=True)
        self.fusion_layer3 = ConvAttention(hidden_dim, head_dim_mul=1.0, querykey=True, convs=False, retAtt=True)
        self.fusion_layer4 = ConvAttention(hidden_dim, head_dim_mul=1.0, querykey=True, convs=False, retAtt=True)

        # Post-attention MLP + norm for each of the 3 streams.
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

        sep_queries = cfg.SEP_QUERIES
        extra_bb = cfg.EXTRA_BB

        # Downsample each template level to a single query vector, then let the
        # 3 (level-2/3/4) queries attend to each other before cross-attention.
        self.query_down = ConvAttention(hidden_dim * (2 if extra_bb else 1), head_dim_mul=1.0, querykey=False,
                                         convs=False, retAtt=False, reduc_channel=0.5 if extra_bb else 1.0)
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.query_encoder = TransformerEncoder(encoder_layer, num_layers=1)
        self.query_pos_embed = nn.Parameter(torch.zeros(1, 3, hidden_dim))
        nn.init.trunc_normal_(self.query_pos_embed, std=.02)

        if sep_queries:
            self.query_down_pos = ConvAttention(hidden_dim, head_dim_mul=1.0, querykey=False, convs=False,
                                                 retAtt=False, reduc_channel=0.5 if extra_bb else 1.0)
            self.fusion_pos_layer2 = ConvAttention(hidden_dim, head_dim_mul=1.0, querykey=True, convs=False, retAtt=True)
            self.fusion_pos_layer3 = ConvAttention(hidden_dim, head_dim_mul=1.0, querykey=True, convs=False, retAtt=True)
            self.fusion_pos_layer4 = ConvAttention(hidden_dim, head_dim_mul=1.0, querykey=True, convs=False, retAtt=True)

    def forward(self, x, z):
        """
        x: search region features, list of 3 tensors (levels 2, 3, 4)
        z: template features, list of 3 tensors (levels 2, 3, 4)
        """
        x_last_three, z_last_three = x, z
        N, C, _, _ = x_last_three[-1].shape

        # --- Build initial per-level template queries, then let them attend to each other. ---
        z2_query = self.query_down(z_last_three[0], mean_query=True).reshape(N, 1, C)
        z3_query = self.query_down(z_last_three[1], mean_query=True).reshape(N, 1, C)
        z4_query = self.query_down(z_last_three[2], mean_query=True).reshape(N, 1, C)

        queries = torch.cat([z2_query, z3_query, z4_query], dim=1) + self.query_pos_embed  # [N, 3, C]
        encoded_queries = self.query_encoder(queries)
        z2_query, z3_query, z4_query = encoded_queries.split(1, dim=1)

        # --- Cross-attention + FFN per level. ---
        attn_matrices = []

        feat_vec_2, attn_2 = self.fusion_layer2.forward_cross(query=z2_query, keyvalue=x_last_three[0])
        out_vec_2 = self.norm1_l2(z2_query + feat_vec_2)
        out_vec_2 = self.norm2_l2(self.mlp_l2(out_vec_2) + out_vec_2)
        attn_matrices.append(attn_2)

        feat_vec_3, attn_3 = self.fusion_layer3.forward_cross(query=z3_query, keyvalue=x_last_three[1])
        out_vec_3 = self.norm1_l3(z3_query + feat_vec_3)
        out_vec_3 = self.norm2_l3(self.mlp_l3(out_vec_3) + out_vec_3)
        attn_matrices.append(attn_3)

        feat_vec_4, attn_4 = self.fusion_layer4.forward_cross(query=z4_query, keyvalue=x_last_three[2])
        out_vec_4 = self.norm1_l4(z4_query + feat_vec_4)
        out_vec_4 = self.norm2_l4(self.mlp_l4(out_vec_4) + out_vec_4)
        attn_matrices.append(attn_4)

        # Concatenate levels (deepest first, matching the head's expected input layout).
        concatenated_features = torch.cat([out_vec_4, out_vec_3, out_vec_2], dim=1).reshape(N, 1, -1)

        # Return the concatenated feature and the deepest level's attention map.
        return concatenated_features, attn_4


class EFFTrackHead(nn.Module):
    """Reads out a box + objectness score from the concatenated query embedding
    produced by EffTrackNeck (attn_matrix is accepted for interface symmetry with
    other head types, but isn't otherwise used)."""

    def __init__(self, cfg=None, hidden_dim=160):
        super().__init__()
        self.cfg = cfg
        self.fmap_size = cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE
        if cfg.MULTI_SCALE:
            self.fmap_size *= 2
        hidfac = 4
        sep_queries = cfg.SEP_QUERIES

        # 3 pyramid levels (from EffTrackNeck) are concatenated into one input vector.
        input_dim = hidden_dim * 3

        if sep_queries:
            self.mlp_score = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * hidfac),
                nn.ReLU(),
                nn.Linear(hidden_dim * hidfac, hidden_dim * hidfac),
                nn.ReLU(),
                nn.Linear(hidden_dim * hidfac, 1),
            )
            self.mlp_pos = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * hidfac),
                nn.ReLU(),
                nn.Linear(hidden_dim * hidfac, hidden_dim * hidfac),
                nn.ReLU(),
                nn.Linear(hidden_dim * hidfac, 4),  # pos offset + size offset
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * hidfac),
                nn.ReLU(),
                nn.Linear(hidden_dim * hidfac, hidden_dim * hidfac),
                nn.ReLU(),
                nn.Linear(hidden_dim * hidfac, 5),  # score + pos offset + size offset
            )

    def forward(self, query_embedding, attn_matrix):
        sep_queries = self.cfg.SEP_QUERIES

        if sep_queries:
            feature_pos, query_embedding = query_embedding

        query_embedding = query_embedding.squeeze(1)  # [N, 1, C] -> [N, C]

        if sep_queries:
            feature_pos = feature_pos.squeeze(1)
            score = self.mlp_score(query_embedding)  # [N, 1]
            bbox = self.mlp_pos(feature_pos)          # [N, 4]
        else:
            bbox_info = self.mlp(query_embedding)  # [N, 5]
            score = bbox_info[:, 0:1]
            pos = bbox_info[:, 1:3]
            size = bbox_info[:, 3:5]
            bbox = torch.cat([pos, size], dim=1)  # [x, y, w, h]

        # Loss expects boxes as [N, 1, 4] and scores as [N, 1].
        return bbox.unsqueeze(1), score


def _build_backbone(cfg, pretrained, training):
    """Builds the backbone module and returns (backbone, hidden_dim)."""
    if "mobilevitv2" in cfg.MODEL.BACKBONE.TYPE:
        width_multiplier = float(cfg.MODEL.BACKBONE.TYPE.split('-')[-1])
        backbone = create_mobilevitv2_backbone(pretrained, width_multiplier,
                                                has_mixed_attn=cfg.MODEL.BACKBONE.MIXED_ATTN, training=training)
        backbone.mixed_attn = bool(cfg.MODEL.BACKBONE.MIXED_ATTN)
        hidden_dim = backbone.model_conf_dict['layer4']['out']
        return backbone, hidden_dim

    if "lowformer" in cfg.MODEL.BACKBONE.TYPE:
        from lib.models.mobilevit_track.lowformer_track import build_lowformer_backbone
        backbone = build_lowformer_backbone(cfg.MODEL.BACKBONE.TYPE.replace("lowformer_", ""), cfg)
        hidden_dim = cfg.MODEL.HEAD.NUM_CHANNELS

        if cfg.MODEL.BACKBONE.EFFTRACK:
            print("Efficient Tracking Model initializing...")
            backbone = EffTrackWrapper(cfg, backbone, training=training, hidden_dim=hidden_dim)
        else:
            backbone = LowFormerWrapper(backbone, training=training, type_emb=cfg.MODEL.TYPE_EMB,
                                         type_embv2=cfg.MODEL.TYPE_EMBV2)
        return backbone, hidden_dim

    raise NotImplementedError


def build_mobilevitv2_track(cfg, settings=None, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE) if cfg.MODEL.PRETRAIN_FILE and training else ''

    backbone, hidden_dim = _build_backbone(cfg, pretrained, training)

    # Build neck module to fuse template and search region features.
    if cfg.MODEL.NECK:
        if cfg.MODEL.NECK.TYPE == "EFFTRACK":
            print("Initializing EffTrack neck...")
            neck = EffTrackNeck(cfg=cfg, hidden_dim=hidden_dim).cuda()

            if training:
                # One-off sanity-check forward pass on random data: builds every
                # submodule's parameters eagerly and fails fast on shape mismatches
                # before the real training loop starts.
                x, z = backbone(torch.randn(2, 3, 256, 256), torch.randn(2, 3, 128, 128))
                x_cuda = [t.cuda() for t in x]
                z_cuda = [t.cuda() for t in z]
                query_embed, attn = neck(x=x_cuda, z=z_cuda)
                print("neck output q:", query_embed.shape, "attn:", attn.shape)
        else:
            neck = build_neck(cfg=cfg, hidden_dim=hidden_dim)
    else:
        neck = nn.Identity()

    if cfg.MODEL.NECK.TYPE == "BN_PWXCORR":
        try:
            in_features = backbone.model_conf_dict['layer4']['out']
        except AttributeError:
            in_features = hidden_dim
        feature_fusor = build_feature_fusor(cfg=cfg, in_features=in_features,
                                             xcorr_out_features=cfg.MODEL.NECK.NUM_CHANNS_POST_XCORR)
    elif cfg.MODEL.NECK.TYPE == "BN_SSAT" or cfg.MODEL.NECK.TYPE == "BN_HSSAT":
        feature_fusor = build_feature_fusor(cfg=cfg, in_features=backbone.model_conf_dict['layer4']['out'],
                                             xcorr_out_features=None)
    elif cfg.MODEL.HEAD.TYPE == "EFFTRACK":
        feature_fusor = None
    else:
        raise NotImplementedError

    # Build head module.
    if cfg.MODEL.HEAD.TYPE == "EFFTRACK":
        box_head = EFFTrackHead(cfg, hidden_dim=hidden_dim).cuda()
        if training:
            with torch.no_grad():
                print(box_head(query_embed, attn))
    else:
        box_head = build_box_head(cfg, cfg.MODEL.HEAD.NUM_CHANNELS)

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
