# LowFormer: Hardware Efficient Design for Convolutional Transformer Backbones
# Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni
# Winter Conference on Applications of Computer Vision (WACV), 2025

import torch, math, sys, os, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchvision.utils as tutils

from utils import build_act
from utils import build_norm
from utils import get_same_padding, list_sum, resize, val2list, val2tuple


__all__ = [
    "ConvLayer",
    "UpSampleLayer",
    "LinearLayer",
    "IdentityLayer",
    "DSConv",
    "MBConv",
    "FusedMBConv",
    "ResBlock",
    "LiteMLA",
    "SMConvLayer",
    "SMConvLayerV2",
    "ConvAttention",
    "LowFormerBlock",
    "ResidualBlock",
    "ResidualConcatLayer",
    "DAGBlock",
    "OpSequential",
]


#################################################################################
#                             Basic Layers                                      #
#################################################################################


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
        transpose=False,
        padding=-1,
    ):
        super(ConvLayer, self).__init__()

        if padding == -1:
            padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        # nn.ConvTranspose2d(input_dim, input_dim, kernel_size=att_stride*(2 if dconvkernel else 1), stride=att_stride, padding=att_stride//2 if dconvkernel else 0 , groups=input_dim)
        if transpose: 
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=stride+2,
                stride=(stride, stride),
                padding=1,
                dilation=(dilation, dilation),
                groups=1,
                bias=use_bias,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=(stride, stride),
                padding=padding,
                dilation=(dilation, dilation),
                groups=groups,
                bias=use_bias,
            )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if not self.norm is None:
            x = self.norm(x)
        if not self.act is None:
            x = self.act(x)
        return x


class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size: int or tuple[int, int] or list[int] or None = None,
        factor=2,
        align_corners=False,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.size is not None and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout=0,
        norm=None,
        act_func=None,
        squeeze_it=False,
    ):
        super(LinearLayer, self).__init__()

        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)
        self.squeeze_it = squeeze_it

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if self.squeeze_it:# or x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if not self.dropout is None:
            x = self.dropout(x)
        x = self.linear(x)
        if not self.norm is None:
            x = self.norm(x)
        if not self.act is None:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


#################################################################################
#                             Basic Blocks                                      #
#################################################################################


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(DSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x



if True:

    class FusedMBConv(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride=1,
            mid_channels=None,
            expand_ratio=6,
            groups=1,
            use_bias=False,
            norm=("bn2d", "bn2d"),
            act_func=("relu6", None),
            fusedgroup=False,
            padding=-1,
        ):
            super().__init__()
            # norm = ("bn2d","bn2d") # TODO REMOVE

            use_bias = val2tuple(use_bias, 2)
            norm = val2tuple(norm, 2)
            act_func = val2tuple(act_func, 2)

            mid_channels = mid_channels or round(in_channels * expand_ratio)

            self.spatial_conv = ConvLayer(
                in_channels,
                mid_channels,
                kernel_size,
                stride,
                groups=2 if fusedgroup and groups== 1 else groups,
                use_bias=use_bias[0],
                norm=norm[0],
                act_func=act_func[0],
                padding=padding,
            )
            self.point_conv = ConvLayer(
                mid_channels,
                out_channels,
                1,
                use_bias=use_bias[1],
                norm=norm[1],
                act_func=act_func[1],
                padding=padding,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.spatial_conv(x)
            x = self.point_conv(x)
            return x
else:
    class FusedMBConv(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride=1,
            mid_channels=None,
            expand_ratio=6,
            groups=1,
            use_bias=False,
            norm=("bn2d", "bn2d"),
            act_func=("relu6", None),
            fusedgroup=False,
        ):
            super().__init__()
            # norm = ("bn2d","bn2d") # TODO REMOVE

            use_bias = val2tuple(use_bias, 2)
            norm = val2tuple(norm, 2)
            act_func = val2tuple(act_func, 2)

            mid_channels = mid_channels or round(in_channels * expand_ratio)
            self.stride = stride
            used_stride = stride * 2

            self.spatial_conv = ConvLayer(
                in_channels,
                mid_channels,
                kernel_size=used_stride+1,
                stride=used_stride,
                groups=groups,
                use_bias=use_bias[0],
                norm=norm[0],
                act_func=act_func[0],
            )

            self.mid_point_conv = ConvLayer(
                mid_channels,
                mid_channels,
                kernel_size=1,
                stride=1,
                use_bias=use_bias[1],
                norm=norm[1],
                act_func=act_func[1],
                transpose=False,
            )
            self.point_conv = ConvLayer(
                mid_channels,
                out_channels,
                kernel_size=1,
                stride=2,
                use_bias=use_bias[1],
                norm=norm[1],
                act_func=act_func[1],
                transpose=True
            )
        
            self.upit = nn.Upsample(scale_factor=2)

            

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.spatial_conv(x)
            # x = self.upit(x)
            x = self.mid_point_conv(x)
            x = self.point_conv(x)
            return x




class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SMConvLayerV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expans=4):
        super(SMConvLayerV2, self).__init__()
        

        self.start = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), nn.SiLU(), nn.Softmax(dim=1))
        self.start2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels*(expans-2), kernel_size=1), nn.BatchNorm2d(in_channels*(expans-2)), nn.SiLU())
        
        self.rest = nn.Sequential(nn.Conv2d(in_channels*expans,in_channels*expans,kernel_size=3, padding=1, stride=stride,groups=in_channels*4), 
                                  nn.BatchNorm2d(in_channels*expans), 
                                  nn.SiLU(),
                                  nn.Conv2d(in_channels=in_channels*expans, out_channels=out_channels, kernel_size=1), 
                                  nn.BatchNorm2d(out_channels), 
                                  nn.SiLU())
        
    # -> pw -> sm
    # -> pw (2c)
    #  c + 2c + c  - > dw -> pw
    def forward(self, x):
        xin = self.start(x)
        xin2 = self.start2(x)
        x = torch.cat([xin,xin2,x],dim=1)
        x = self.rest(x)
        return x

class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        grouping=1,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
        bb_smbconv=False,
        new_smbconv=False,
    ):
        super(MBConv, self).__init__()
        # norm = (None, None,"bn2d") # TODO REMOVE
        

        self.new_smbconv = new_smbconv
        self.bb_smbconv = bb_smbconv
        self.stride = stride
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.in_channels = in_channels
        
        if new_smbconv:
            # self.point_conv = nn.Sequential(ConvLayer(in_channels, in_channels, kernel_size=1, stride=1, norm=norm[0], act_func=act_func[0], use_bias=use_bias[0]),nn.Softmax(dim=1))
            
            self.point_conv = ConvLayer(in_channels, in_channels*(expand_ratio-1), kernel_size=1, norm=norm[0], act_func=act_func[0], use_bias=use_bias[0])
            self.sm = nn.Softmax(dim=1)
            
            self.rest = nn.Sequential( ConvLayer(in_channels*expand_ratio,in_channels*expand_ratio,
                                                 kernel_size=3, stride=stride,
                                                 groups=in_channels*expand_ratio, 
                                                 norm=norm[1], act_func=act_func[1], use_bias=use_bias[1]),
                                      ConvLayer(in_channels*expand_ratio,out_channels,
                                                kernel_size=1, 
                                                norm=norm[2], act_func=act_func[2], use_bias=use_bias[2])
                                      )
            
        else:
            self.inverted_conv = ConvLayer(
                in_channels,
                mid_channels,
                1,
                stride=1,
                groups=grouping, # TODO
                norm=norm[0],
                act_func=act_func[0],
                use_bias=use_bias[0],
            )
            self.depth_conv = ConvLayer(
                mid_channels,
                mid_channels,
                kernel_size,
                stride=stride,
                groups=mid_channels,
                norm=norm[1],
                act_func=act_func[1],
                use_bias=use_bias[1],
            )

            if (self.bb_smbconv and not self.stride>1) or self.new_smbconv:
                self.sm = nn.Softmax(dim=1)

            self.point_conv = ConvLayer(
                mid_channels,
                out_channels,
                1,
                groups=grouping, # TODO
                norm=norm[2],
                act_func=act_func[2],
                use_bias=use_bias[2],
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.new_smbconv:
            assert False
            xin = self.point_conv(x)
            xin[:,:self.in_channels,:,:] = self.sm(xin[:,:self.in_channels,:,:])
            x = torch.cat([xin,x],dim=1)
            x = self.rest(x)
            return x
        else:
            x = self.inverted_conv(x)

            if self.new_smbconv:
                assert False
                x[:,:self.in_channels,:,:] = self.sm(x[:, :self.in_channels, :, :])
                # print("shapes:",x.shape,x[:,:self.in_channels,:,:].shape)

            if self.bb_smbconv and not self.stride>1:
                assert False
                x1 = self.depth_conv(x)
                x1 = self.sm(x1)
                x1 = x1 / (torch.std(x1,dim=(0,1),keepdim=True) * 2)
                x = x + x1
            else:
                x = self.depth_conv(x)

            x = self.point_conv(x)
            return x
    


class SMConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        use_bias=False,
        norm="bn2d",
        act_func="relu",
        smconv_dw=False,
        bb_nosm=False,
    ):
        super(SMConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation
        self.smconv_dw = smconv_dw


        if self.smconv_dw:
            stage = []
            # pw
            stage.append(ConvLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=1,norm=norm, act_func=act_func))
            if not bb_nosm:
                stage.append(nn.Softmax(dim=1))
            stage = nn.Sequential(*stage)
            pwin = ResidualConcatLayer(main=stage,shortcut=IdentityLayer(),dim=1)
            dwconv = ConvLayer(in_channels*2,in_channels*2,kernel_size=3,stride=stride,groups=in_channels*2,norm=norm,act_func=act_func)
            pwconv = ConvLayer(in_channels*2,out_channels,kernel_size=1,norm=norm,act_func=act_func)
            self.total = nn.Sequential(pwin, dwconv, pwconv)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=1,
                padding=padding,
                dilation=(dilation, dilation),
                groups=in_channels,
                bias=use_bias,
            )
            norm = build_norm(norm, num_features=out_channels)
            act = build_act(act_func)

            self.conv = nn.Sequential(self.conv, norm, act)
            if bb_nosm:
                self.sm = nn.Identity()
            else:
                self.sm = nn.Softmax(dim=1)
            self.convpw = nn.Conv2d(out_channels*2,out_channels, kernel_size=1, stride=1, padding=0,bias=use_bias)
            if False:
                self.convpw = nn.Sequential(self.convpw, build_norm(norm, num_features=out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.smconv_dw:
            x = self.total(x)
        else:
            x1 = self.conv(x)
            x1 = self.sm(x1)
            # x = x1
            x = self.convpw(torch.cat([x,x1],dim=1))
        return x


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int or None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    # @autocast(enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (
            qkv[..., 0 : self.dim],
            qkv[..., self.dim : 2 * self.dim],
            qkv[..., 2 * self.dim :],
        )

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1.0)
        # print("B/H/W/DIM",B,H,W,self.dim)  # 128 14 14 16     
        # print("transk/v",trans_k.shape, v.shape) # [128, 16, 16, 196] [128, 16, 196, 17]
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        # print("q/kv/out",q.shape, kv.shape, out.shape)# q/kv/out torch.Size([128, 16, 196, 16]) torch.Size([128, 16, 16, 17]) torch.Size([128, 16, 196, 17])
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = torch.transpose(out, -1, -2) # -> 128, 16, 16, 196]
        out = torch.reshape(out, (B, -1, H, W)) # -> 128, 16*16, 14, 14
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        out = self.relu_linear_att(multi_scale_qkv)
        out = self.proj(out)

        return out
    

class SDALayer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.iden = nn.Identity()
    
    def scaled_dot_product(self, q, k, v):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        
        if True: # test without softmax!
            attention = F.softmax(attn_logits, dim=-1)
        else:
            attention = attn_logits

        values = torch.matmul(attention, v)
        return values#, attention
    
    def forward(self, q, k, v) -> torch.Tensor:
        return self.scaled_dot_product(q,k,v)


class ConvAttention(nn.Module):
    def __init__(self, input_dim, full=True, head_dim_mul=1.0, att_stride=4, att_kernel=7, dconvkernel=True, sha=False, actit=False, pwopt=True, mscale=False, sdalayer=True, fuseconv=False, fuseconvall=False, newhdim=False, notransp=False):
        super().__init__()
        # assert input_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads. Problem embed: %d | num_heads %d" % (input_dim, num_heads)

        self.head_dim_mul = head_dim_mul # TODO if mscale, headdimmul = 0.5
        self.newhdim = newhdim
        if input_dim < 50:
            self.newhdim=False
        if mscale:
            self.head_dim_mul = 0.35
        if self.newhdim:
            self.head_dim_mul = 0.5
        self.mscale = mscale
        self.pwopt = pwopt
        self.sha = sha
        self.num_heads = int(max(1,(input_dim*self.head_dim_mul)//30))
        self.input_dim = input_dim
        self.head_dim = int((input_dim // self.num_heads) * self.head_dim_mul)
        self.full = full
        self.num_keys = (3 if full else 2)
        self.att_stride = att_stride
        self.notransp = notransp

        assert full
        assert not self.newhdim
        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        # self.qkv_proj = nn.Linear(input_dim, self.num_keys * self.head_dim * self.num_heads)
        # self.q_proj = nn.Linear(input_dim, self.head_dim * self.num_heads)
        # self.k_proj = nn.Linear(input_dim, self.head_dim * self.num_heads)
        # self.v_proj = nn.Linear(input_dim, self.head_dim * self.num_heads)
        total_dim = int(self.head_dim * self.num_heads * self.num_keys)
        if sha:
            total_dim = int((0.3 * total_dim //3 + 1 ) * 3)
        assert not (sha and mscale), "sha and mscale don't work together yet!"

        if self.mscale:
            self.conv_proj = [nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=att_kernel, stride=att_stride, padding=att_kernel//2, groups=input_dim, bias=False), nn.BatchNorm2d(input_dim))]
            if not att_kernel == 3:
                self.conv_proj += [nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=att_kernel-2, stride=att_stride, padding=att_kernel//2 - 1, groups=input_dim, bias=False), nn.BatchNorm2d(input_dim))]
            else:
                self.conv_proj += [nn.Identity()]
                # self.conv_proj += [nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=att_kernel+4, stride=att_stride, padding=att_kernel//2 +2, groups=input_dim, bias=False), nn.BatchNorm2d(input_dim))]
            
            self.conv_proj += [nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=att_kernel+2, stride=att_stride, padding=att_kernel//2 + 1, groups=input_dim, bias=False), nn.BatchNorm2d(input_dim))]
            self.conv_proj = nn.ModuleList(self.conv_proj)

            self.pwise = [nn.Conv2d(input_dim, total_dim, kernel_size=1, stride=1, padding=0, bias=False) for i in range(len(self.conv_proj))]
            self.pwise = nn.ModuleList(self.pwise)
        else:
            self.conv_proj = nn.Conv2d(input_dim, input_dim, kernel_size=att_kernel, stride=att_stride, padding=att_kernel//2, groups=input_dim, bias=False)
            if head_dim_mul < 1 or True: 
                self.conv_proj = nn.Sequential(self.conv_proj, nn.BatchNorm2d(input_dim))
                if actit:
                    self.conv_proj = nn.Sequential(self.conv_proj, nn.Hardswish())
                self.pwise = nn.Sequential(nn.Conv2d(input_dim, total_dim, kernel_size=1, stride=1, padding=0, bias=False))
            if fuseconvall and input_dim<256: 
                self.conv_proj = nn.Conv2d(input_dim, total_dim, kernel_size=att_kernel, stride=att_stride, padding=att_kernel//2, bias=False)
                self.pwise = nn.Identity()
        
        
        
        
        if sdalayer:
            self.sda = SDALayer()
        else:
            self.sda = None
        
        self.o_proj_inpdim = self.head_dim * self.num_heads* (len(self.conv_proj) if self.mscale else 1)
        if self.newhdim:
            self.o_proj_inpdim = self.input_dim
        
        self.o_proj = nn.Linear(self.o_proj_inpdim, input_dim)

        if self.pwopt:
            self.o_proj = nn.Conv2d(self.o_proj_inpdim, input_dim, kernel_size=1,stride=1,padding=0)

        
        self.upsampling = nn.ConvTranspose2d(input_dim, input_dim, kernel_size=att_stride*(2 if dconvkernel else 1), stride=att_stride, padding=att_stride//2 if dconvkernel else 0 , groups=input_dim)
        if att_stride == 1 or notransp:
            self.upsampling = nn.ConvTranspose2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim)
        if actit:
            self.upsampling = nn.Sequential(self.upsampling, nn.BatchNorm2d(input_dim), nn.Hardswish())

        if fuseconv:
            self.o_proj = nn.Identity()
            if att_stride == 1 or notransp:
                self.upsampling = nn.ConvTranspose2d(self.o_proj_inpdim, input_dim,kernel_size=3, stride=1, padding=1)
            else:
                self.upsampling = nn.ConvTranspose2d(self.o_proj_inpdim, input_dim, kernel_size=att_stride*(2 if dconvkernel else 1), stride=att_stride, padding=att_stride//2 if dconvkernel else 0 )
            
    # @torch.jit.ignore
    def scaled_dot_product(self, q, k, v):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        # if mask is not None:
        #     attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        #     assert False, "no masks included!"
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, x):
        N, C, H, W = x.size()
        # print("before:",x.shape)

        # Downsampling
        # [N,C, H, W] -> [N,c,h,w]

        # def run_att(xin, conv_proj, pwise):
        #     xout = conv_proj(xin)
        #     xout = pwise(xout)
            
        #     # Transformer part
        #     N, c, h, w = xout.size()
        #     # assert H / self.att_stride == h, "H:%d h:%f H/h:%f" %(H,h,H/h)

        #     # print(x.shape, self.num_heads, self.num_keys*self.head_dim, h*w)
        #     # Separate Q, K, V from linear output
        #     if self.sha:
        #         qkv = xout.reshape(N,1, c, h*w)
        #     else:
        #         qkv = xout.reshape(N, self.num_heads, self.num_keys*self.head_dim, h*w)

            
            
        #     qkv = qkv.permute(0, 1, 3, 2) # [N, Head, SeqLen, Dims]
            
        #     if self.newhdim:
        #         qt, kt= int(self.input_dim*3*self.head_dim_mul / (self.num_heads*6)), int(self.input_dim*3*self.head_dim_mul / (self.num_heads*6))*2 #int(self.input_dim*3*0.5 * self.head_dim_mul / self.num_heads)#, int(self.input_dim*3* 0.5* self.head_dim_mul / self.num_heads)
        #         # kt = qkv.shape[-1] - self.o_proj_inpdim
        #         q, k, v = qkv[:,:,:,:qt], qkv[:,:,:,qt:kt], qkv[:,:,:,kt:]
        #         # print(self.head_dim ,self.num_heads ,self.num_keys,qt,kt)
        #         # assert q.shape[-1] + k.shape[-1] + v.shape[-1] == qkv.shape[-1], str(qkv.shape) + "|| %d %d %d %d %f" % (q.shape[-1],k.shape[-1],v.shape[-1], self.input_dim, self.head_dim_mul)
        #     elif self.full:
        #         q, k, v = qkv.chunk(self.num_keys, dim=-1) # [N,Head,Seqlen,dims/3]
        #     else:
        #         q, v = qkv.chunk(self.num_keys, dim=-1) # [N,Head,Seqlen,dims/2]
        #         k = q / LA.vector_norm(q,dim=(1,3),keepdim=True)  # [N, head, L, embed] normalize q vector
        #     # assert (q.shape[-1] == k.shape[-1] and k.shape[-1] == v.shape[-1]) or (q.shape[-1] == k.shape[-1] and self.newhdim), "shapes of q, k and v don't align!"
            
   
        #     # Determine value outputs
        #     if not self.sda is None:
        #         values, attention = self.sda(q,k,v)
        #     else:
        #         values, attention = self.scaled_dot_product(q, k, v, mask=None) # [N,head,Seqlen,dims/3]
        #     return values, c, h, w

        if self.mscale:
            assert False
            # all_values = []
            # for i in range(len(self.conv_proj)):
            #     temp, c, h, w = run_att(x, self.conv_proj[i], self.pwise[i])
            #     all_values.append(temp)
            # values = torch.cat(all_values,dim=1)
        else:
            # values, c, h, w = run_att(x, self.conv_proj, self.pwise)
            
            xout = self.conv_proj(x)
            xout = self.pwise(xout)
            # Transformer part
            N, c, h, w = xout.size()
            # assert H / self.att_stride == h, "H:%d h:%f H/h:%f" %(H,h,H/h)

            # print(x.shape, self.num_heads, self.num_keys*self.head_dim, h*w)
            # Separate Q, K, V from linear output
            if self.sha:
                qkv = xout.reshape(N,1, c, h*w)
            else:
                qkv = xout.reshape(N, self.num_heads, self.num_keys*self.head_dim, h*w)

            
            
            qkv = qkv.permute(0, 1, 3, 2) # [N, Head, SeqLen, Dims]
            #########################
            # if self.newhdim and False:
            #     qt, kt= int(self.input_dim*3*self.head_dim_mul / (self.num_heads*6)), int(self.input_dim*3*self.head_dim_mul / (self.num_heads*6))*2 #int(self.input_dim*3*0.5 * self.head_dim_mul / self.num_heads)#, int(self.input_dim*3* 0.5* self.head_dim_mul / self.num_heads)
            #     # kt = qkv.shape[-1] - self.o_proj_inpdim
            #     q, k, v = qkv[:,:,:,:qt], qkv[:,:,:,qt:kt], qkv[:,:,:,kt:]
            #     # print(self.head_dim ,self.num_heads ,self.num_keys,qt,kt)
            #     # assert q.shape[-1] + k.shape[-1] + v.shape[-1] == qkv.shape[-1], str(qkv.shape) + "|| %d %d %d %d %f" % (q.shape[-1],k.shape[-1],v.shape[-1], self.input_dim, self.head_dim_mul)
            # elif self.full and self.num_keys == 3:
            #     q, k, v = qkv.chunk(self.num_keys, dim=-1) # [N,Head,Seqlen,dims/3]
            #################
            q, k, v = qkv.chunk(3, dim=-1)    
            
            # else:
            #     q, v = qkv.chunk(self.num_keys, dim=-1) # [N,Head,Seqlen,dims/2]
            #     k = q / LA.vector_norm(q,dim=(1,3),keepdim=True)  # [N, head, L, embed] normalize q vector
            # assert (q.shape[-1] == k.shape[-1] and k.shape[-1] == v.shape[-1]) or (q.shape[-1] == k.shape[-1] and self.newhdim), "shapes of q, k and v don't align!"
            

            ## Determine value outputs
            # remove att
            if False:
                values = v
            else:
                if not self.sda is None:
                    values = self.sda(q,k,v)
                else:
                    values, attention = self.scaled_dot_product(q, k, v) # [N,head,Seqlen,dims/3]
            
            
    
  
  
  

        if self.sha:
            values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
            values = values.reshape(N, h,w, -1).permute(0,3,1,2)
            values = torch.cat([values,x[:,:x.shape[1] - values.shape[1]]],dim=1).permute(0,2,3,1)
            o = self.o_proj(values).reshape(N, h, w, self.input_dim).permute(0,3,1,2)
        else:
            if self.pwopt:
                # print(values.permute(0,1,3,2).shape)
                o = self.o_proj(values.permute(0,1,3,2).reshape(N,self.o_proj_inpdim,h,w)) #[N,C,h,w]
            else:
                values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
                values = values.reshape(N, h*w, self.o_proj_inpdim)
                o = self.o_proj(values).reshape(N, h, w, self.input_dim).permute(0,3,1,2)

        # Upsampling
        if self.att_stride > 1 and self.notransp:# and H != h:
            o = F.interpolate(o, size=(H,W), mode='nearest')
            o = self.upsampling(o)
            # assert False
        else:
            o = self.upsampling(o)
        
        return o[:N,:C,:H,:W]


class ConvCombAttention(nn.Module):
    def __init__(self, input_dim, full=True, head_dim_mul=1.0, att_stride=4, att_kernel=7, dconvkernel=True, multcut=False, sha=False, actit=False, mscale=False):
        super().__init__()
        # assert input_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads. Problem embed: %d | num_heads %d" % (input_dim, num_heads)
        # input_dim = chs
        chs = input_dim
        self.multcut = multcut
        self.num_heads_sp = int(max(1,(input_dim*head_dim_mul)//30))
        self.input_dim = input_dim
        self.head_dim_sp = int((input_dim // self.num_heads_sp) * head_dim_mul)
        self.full = full
        self.num_keys = (3 if full else 2)
        self.att_stride = att_stride
        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        # self.qkv_proj = nn.Linear(input_dim, self.num_keys * self.head_dim * self.num_heads)
        # self.q_proj = nn.Linear(input_dim, self.head_dim * self.num_heads)
        # self.k_proj = nn.Linear(input_dim, self.head_dim * self.num_heads)
        # self.v_proj = nn.Linear(input_dim, self.head_dim * self.num_heads)
        total_dim = int(self.head_dim_sp * self.num_heads_sp * self.num_keys)
        
        # start
        self.downsampling = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=att_kernel, stride=att_stride, padding=att_kernel//2, groups=input_dim, bias=False), nn.BatchNorm2d(input_dim))
        
        # channel
        self.pwise_ch = nn.Conv2d(input_dim, input_dim*self.num_keys, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_ch = nn.Conv2d(input_dim, input_dim, kernel_size=1,stride=1,padding=0)
        self.LN_ch = nn.GroupNorm(num_groups=1, num_channels=input_dim)

        # spatial
        self.pwise_sp = nn.Conv2d(input_dim, total_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_sp = nn.Conv2d(self.head_dim_sp*self.num_heads_sp, input_dim, kernel_size=1,stride=1,padding=0)
        self.LN_sp = nn.GroupNorm(num_groups=1, num_channels=input_dim)

        if multcut:
            self.mulskip_extrac = nn.Sequential(
                    nn.Conv2d(chs, chs, kernel_size=3,padding=1, groups=chs),
                    nn.BatchNorm2d(chs),
                    nn.SiLU(),
                    nn.Conv2d(chs, chs, kernel_size=1,padding=0),
                    nn.BatchNorm2d(chs),
                    nn.SiLU(),
                    nn.Conv2d(chs, chs, kernel_size=3,padding=1, groups=chs),
                    # nn.BatchNorm2d(chs),
                    # MBConv(chs, chs, expans=1, use_se=False),
                    nn.Sigmoid()
                )


        self.upsampling = nn.ConvTranspose2d(input_dim, input_dim, kernel_size=att_stride*(2 if dconvkernel else 1), stride=att_stride, padding=att_stride//2 if dconvkernel else 0 , groups=input_dim)
        if att_stride == 1:
            self.upsampling = nn.ConvTranspose2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim)
          
          
    def scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
            assert False, "no masks included!"
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, x, mask=None):
        # print("before:",x.shape)
        
        # Downsampling
        # [N,C, H, W] -> [N,c,h,w]
        x_down = self.downsampling(x)
        N, C, H, W = x_down.size()

        ## CHANNEL
        xout = self.LN_ch(x_down)
        xout = self.pwise_ch(xout) # [N,3C,H,W]
        xout = xout.reshape(N, 1, self.num_keys*C, H*W)
        # SDA in:[N,Head,Seqlen,dims/3]
        q, k, v = xout.chunk(self.num_keys, dim=2)
        values, _ = self.scaled_dot_product(q,k,v)
        values = values.reshape(N,C,H,W)
        values = self.out_ch(values)
        x_down = x_down + values

        ## SPATIAL
        xout = self.LN_sp(x_down)
        xout = self.pwise_sp(xout)

        qkv = xout.reshape(N, self.num_heads_sp, self.num_keys*self.head_dim_sp, H*W)
        qkv = qkv.permute(0, 1, 3, 2) # [N, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(self.num_keys, dim=-1) # [N,Head,Seqlen,dims/3]
        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v, mask=None) # [N,head,Seqlen,dims/3]
        
        values = self.out_sp(values.permute(0,1,3,2).reshape(N,self.head_dim_sp*self.num_heads_sp,H,W)) #[N,C,h,w]
        
        # Residual from before LN
        x_down = x_down + values

        ## Upsampling
        if False and H != h:
            o = F.interpolate(x_down, size=(H,W), mode='nearest')
        else:
            o = self.upsampling(x_down)
        
        if self.multcut:
            o = o * self.mulskip_extrac(x)
        
        x = x + o
        return x


class LowFormerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        scales=(5,),
        norm="bn2d",
        act_func="hswish",
        fuseconv=False,
        fuseconvall=False,
        newhdim=False,
        bb_convattention=False,
        bb_convin2=False,
        mscale=False,
        grouping=1,
        convcomb=False,
        actit=False,
        head_dim_mul=False,
        stage_num=-1,
        bb_smbconv=False,
        sha=False,
        add_smconv=False,
        smconv_pos="befAtt",
        without_mbconv=False,
        smconv_dw=False,
        bb_nosm=False,
        only_smconv=False,
        old_way=False,
        new_smbconv=False,
        old_way_norm=False,
        just_unfused=False,
        noattention=False,
        nostrideatt=False,
        mlpremoved=False,
    ):
        super(LowFormerBlock, self).__init__()
        self.old_way = old_way
        self.add_smconv = add_smconv
        self.smconv_pos = smconv_pos
        # ConvAttention(input_dim=in_channels, num_heads=max(1,in_channels//30), att_stride=2 if stage_num==4 else 1, att_kernel=5 if stage_num==4 else 3)
        block = SMConvLayer(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        norm=norm,
                        act_func=act_func,
                        smconv_dw=smconv_dw,
                        bb_nosm=bb_nosm,
                    )
        smconv_module =  ResidualBlock(block, IdentityLayer())


        context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
                scales=scales,
            ),
            IdentityLayer(),
        )
        if bb_convattention:# and not noattention:
            # Params
            attvers = ConvCombAttention if convcomb else ConvAttention 
            att_stride = 2 if stage_num==3 or (bb_convin2 and stage_num < 3) else (4 if stage_num < 3 else 1)
            if nostrideatt:
                att_stride = 1
            att_kernel = 5 if stage_num==3 or (bb_convin2 and stage_num < 3) else (7 if stage_num < 3 else 3)
            # att_stride = 1 if fuseconv else att_stride # TODO changed
            mlpexpans =  4 if fuseconv else 2 # TODO changed
            

            # (in_channels//3)*3 + 1
            # block
            block = attvers(input_dim=in_channels,
            att_stride=att_stride,            
            att_kernel=att_kernel, head_dim_mul=0.5 if in_channels >80 and head_dim_mul else 1.0, sha=sha, actit=actit, mscale=mscale, fuseconv=fuseconv and not just_unfused, fuseconvall=fuseconvall and not just_unfused, newhdim=newhdim)
            if noattention:
                context_module = nn.Identity()
            elif bb_convin2 and not mlpremoved: 
                context_module = ResidualBlock(nn.Sequential(nn.GroupNorm(1,in_channels),block),IdentityLayer())
            else:
                context_module = ResidualBlock(block,IdentityLayer())
            if bb_convin2 and not mlpremoved:
                mlp = nn.Sequential(nn.GroupNorm(1,in_channels),
                                                                 nn.Conv2d(in_channels,in_channels*mlpexpans,kernel_size=1),
                                                                 nn.GELU(),
                                                                 nn.Conv2d(in_channels*mlpexpans,in_channels,kernel_size=1))
                # mlp = nn.Identity()
                context_module = nn.Sequential(context_module, 
                                               ResidualBlock(
                                                   mlp,
                                                   IdentityLayer()))
        
        # FUSE MBCONV
        if fuseconv and in_channels < 256 and not just_unfused:
            local_module = FusedMBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                use_bias=(True, False),
                norm=(None, norm) if old_way_norm else norm,
                act_func=(act_func, None),
            )
        elif fuseconvall and not just_unfused: #TODO
            local_module = FusedMBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                use_bias=(True, False),
                norm=(None, norm) if old_way_norm else norm, 
                act_func=(act_func, None),
            )
        else:
            local_module = MBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                grouping=grouping,
                use_bias=(True, True, False),
                norm=(None, None, norm),
                act_func=(act_func, act_func, None),
                bb_smbconv=bb_smbconv,
                new_smbconv=new_smbconv,
            )
        if only_smconv:
            local_module = SMConvLayer(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        norm=norm,
                        act_func=act_func,
                        smconv_dw=smconv_dw,
                    )
        local_module = ResidualBlock(local_module, IdentityLayer())
        if without_mbconv:
            local_module = IdentityLayer()
        
        # if noattention:
        #     context_module = nn.Identity()
            
            
        if old_way:
            self.context_module = context_module
            self.local_module = local_module
        else:
            self.total = nn.Sequential(context_module, local_module)
            if self.add_smconv:
                if self.smconv_pos == "befAtt":
                    self.total = nn.Sequential(smconv_module, context_module, local_module)
                elif smconv_pos == "aftAtt":
                    self.total = nn.Sequential(context_module, smconv_module, local_module)
                elif smconv_pos == "aftMBconv":
                    self.total = nn.Sequential(context_module, local_module, smconv_module)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # befAtt|aftAtt|aftMBconv
        # x = self.context_module(x)
        # x = self.local_module(x)
        if self.old_way:
            assert False
            x = self.context_module(x)
            x = self.local_module(x)
        else:
            x = self.total(x)
        return x


#################################################################################
#                             Functional Blocks                                 #
#################################################################################


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: nn.Module or None,
        shortcut: nn.Module or None,
        post_act=None,
        pre_norm: nn.Module or None = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            # print(self.forward_main(x).shape,self.shortcut(x).shape)
            res = self.forward_main(x) + self.shortcut(x)
            if not self.post_act is None:
                res = self.post_act(res)
        return res


class ResidualConcatLayer(nn.Module):
    def __init__(self, main: nn.Module or None, shortcut: nn.Module or None ,dim=0, ):
        super(ResidualConcatLayer, self).__init__()
        self.dim = dim
        self.main = main
        self.shortcut = shortcut
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.main(x),self.shortcut(x)],dim=self.dim)

class DAGBlock(nn.Module):
    def __init__(
        self,
        inputs: dict[str, nn.Module],
        merge: str,
        post_input: nn.Module or None,
        middle: nn.Module,
        outputs: dict[str, nn.Module],
    ):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.concat(feat, dim=1)
        else:
            raise NotImplementedError
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


class OpSequential(nn.Module):
    def __init__(self, op_list: list[nn.Module or None]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


def test_channel_attention():
    
    x = torch.randn((100,60,56,56))
    net = ConvAttention(input_dim=60, num_heads=5, att_stride=4, att_kernel=7, dconvkernel=True, sha=True)
    for i in range(50):
        out = net(x)
    print(out.shape)

if __name__ == '__main__':
    test_channel_attention()