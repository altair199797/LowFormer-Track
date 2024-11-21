# LowFormer: Hardware Efficient Design for Convolutional Transformer Backbones
# Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni
# Winter Conference on Applications of Computer Vision (WACV), 2025

import torch, os, sys
import torch.nn as nn

from Wymodelgetter.ops import (
    ConvLayer,
    DSConv,
    LowFormerBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
)
from Wymodelgetter.utils import build_kwargs_from_config
import torchvision.utils as tutils
from typing import Tuple, List, Dict


__all__ = [
    "LowFormerBackbone",
    "efficientvit_backbone_b0",
    "lowformer_backbone_b1",
    "efficientvit_backbone_b2",
    "efficientvit_backbone_b3",
    "EfficientViTLargeBackbone",
    "efficientvit_backbone_l0",
    "efficientvit_backbone_l1",
    "efficientvit_backbone_l2",
    "efficientvit_backbone_l3",
]


class LowFormerBackbone(nn.Module):
    def __init__(
        self,
        width_list: List[int],
        depth_list: List[int],
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
        bb_convattention=False,
        bb_convin2=False,
        fastit=False,
        fastitv2=False,
        fastitv3=False,
        fastitv4=False,
        smallit=False,
        huge_model=False,
        fuse_conv_all=False,
        newfastit=False,
        bigit=False,
        mscale=False,
        grouping = 1,
        head_dim_mul=False,
        nohdimmul=False,
        actit=False,
        without_mbconv=False,
        just_unfused=False,
        mlpremoved=False,
        nostrideatt=False,
        noattention=False,
        sha=False,
        old_way_norm=False, ########### POST WAC
        fusedgroup=False,
    ) -> None:
        super().__init__()

        
        self.width_list = []
        self.old_way_norm = old_way_norm
        stage_num = 0
        self.max_stage_id = 8
        self.ret_stages = 1
        
        ## STEM stage
        self.input_stem = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for temind in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=4 if huge_model else 1,
                fusedmbconv=fastit,
                grouping=grouping,
                norm=norm,
                act_func=act_func,
                bb_smbconv=self.bb_smbconv == "all",
                just_unfused=just_unfused,
                self=self,
                fusedgroup=fusedgroup,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
            
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        stage_num += 1
    
    
        ## Middle stages 
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                # start block with SMconvlayer
                block = self.build_local_block(
                        in_channels=in_channels,
                        out_channels=w,
                        stride=stride,
                        expand_ratio=6 if stride == 2 and (bigit or newfastit or huge_model) else expand_ratio,# if stride==2 else 2, # TODO CHANGE!!!
                        fusedmbconv=fastit,
                        grouping=grouping,
                        norm=norm,
                        act_func=act_func,
                        self=self,
                        just_unfused=just_unfused,
                    )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                    
                    
                if not without_mbconv or stride != 1:
                    stage.append(block)
                    # in_channels = w

                in_channels = w

            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
            stage_num += 1


        ## Attention stage
        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=2 if smallit else (6 if bigit or (huge_model and stage_num<4) else expand_ratio),
                fusedmbconv=fastit and (not huge_model or stage_num<5) ,
                grouping=grouping,
                norm=norm,
                act_func=act_func,
                fewer_norm=False or old_way_norm,
                self=self,
                just_unfused=just_unfused,
                fusedgroup=fusedgroup,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w
           

            # input_dim, num_heads, full=True, head_dim_mul=1.0, att_stride=4, att_kernel=7, dconvkernel=True
            for _ in range(d):
                stage.append(
                    LowFormerBlock(
                        in_channels=in_channels,
                        expand_ratio=2 if fastitv3 or smallit else expand_ratio, #TODO
                        norm=norm,
                        act_func=act_func,
                        bb_convattention=bb_convattention,
                        fuseconv=fastit,
                        fuseconvall=fastitv2 or fuse_conv_all,
                        newhdim=fastitv4,
                        bb_convin2=bb_convin2,
                        mscale=mscale,
                        grouping=grouping,
                        stage_num=stage_num,
                        sha=sha and stage_num > 2,
                        actit=actit,
                        head_dim_mul=(head_dim_mul or fastit) and not nohdimmul, #TODO
                        just_unfused=just_unfused,
                        noattention=noattention,
                        nostrideatt=nostrideatt,
                        mlpremoved=mlpremoved,
                    )
                )

            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
            stage_num += 1

        self.stages = nn.ModuleList(self.stages)



    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        grouping: int = 1,
        fewer_norm: bool = False,
        fusedmbconv: bool = False,
        just_unfused: bool = False,
        self = None,
        fusedgroup=False,
    ) -> nn.Module:
        if expand_ratio == 1:
            ## DEEP Seperable Conv
            if fusedmbconv:
                block = FusedMBConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        expand_ratio=2,
                        use_bias=(True, False) if fewer_norm else False,
                        norm= (None, norm) if fewer_norm and self.old_way_norm else norm,
                        act_func=(act_func, None),
                        fusedgroup=fusedgroup,
                )
                if just_unfused:
                    block = MBConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        expand_ratio=2,
                        grouping=grouping,
                        use_bias=(True, True, False) if fewer_norm else False,
                        norm=(None, None, norm) if fewer_norm or True else norm,
                        act_func=(act_func, act_func, None),
                    )
            else:
                block = DSConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    use_bias=(True, False) if fewer_norm else False,
                    norm=(None, norm) if fewer_norm else norm,
                    act_func=(act_func, None),
                )
        else: ## ONLY SMCONV
            if fusedmbconv and not just_unfused:
                    block = FusedMBConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        expand_ratio=expand_ratio,
                        use_bias=(True, False) if fewer_norm else False,
                        norm=(None, norm) if fewer_norm and self.old_way_norm else norm,
                        act_func=(act_func, None),
                        fusedgroup=fusedgroup,
                    )
            else:
            ## LOCAL BLOCK
                block = MBConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    grouping=grouping,
                    use_bias=(True, True, False) if fewer_norm else False,
                    norm=(None, None, norm) if fewer_norm or True else norm,
                    act_func=(act_func, act_func, None),
                )
        return block

    def return_stages(self, n):
        assert n>0, n    
        self.ret_stages = n

    def cut_stages(self):
        self.stages = self.stages[:self.max_stage_id]
    
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        # if x.shape[0] > 5:
        #     if not os.path.exists("dump_data"):
        #         os.makedirs("dump_data")
        #     print("in bb:",x.shape)
        #     def dropmap(data, location="dump_data/att_"):
        #         tn, th, thw,_ = data.shape 
        #         data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
        #         for i in range(tn):
        #             fold = location+str(i)+"/"
        #             if not os.path.exists(fold):
        #                 os.makedirs(fold)
        #             tutils.save_image(data[i,:],fold+"img.png")
        #     dropmap(x)

        if False:
            output_dict = {}#{"input": x}
            output_dict["stage0"] = x = self.input_stem(x)
            temp = 0
            for stage_id, stage in enumerate(self.stages, start=1):
                if stage_id > self.max_stage_id:
                    break
                output_dict["stage%d" % stage_id] = x = stage(x)
                temp = stage_id
            # output_dict["stage_final"] = self.stages[-1](x)
            output_dict["stage_final"] = output_dict.pop("stage%d" % temp)
            return output_dict["stage_final"]
        else:
            if self.ret_stages == 1:
                x = self.input_stem(x)
                for stage_id, stage in enumerate(self.stages, start=1):
                    x = stage(x)
                    
                return {4:x}
            elif True:
                x = self.input_stem(x)
                out_dict = {}
                outputs = [x]
                for stage_id, stage in enumerate(self.stages, start=1):
                    x = stage(x)
                    outputs.append(x)
                    
                    # if stage_id >= len(self.stages) - (self.ret_stages-1):
                    #     out_dict[int(stage_id)] = x
                
                ret_dict =  {(ind+len(self.stages) - (self.ret_stages - 1)):i for ind, i in enumerate(outputs[len(self.stages) - (self.ret_stages - 1):]) }
                # print(self.ret_stages, len(self.stages))
                # print({key:(value.shape if not value is None else None) for key,value in ret_dict.items()})
                return ret_dict
            else:
                x = self.input_stem(x)
                out_dict = {}
                for stage_id, stage in enumerate(self.stages, start=1):
                    if stage_id > self.max_stage_id:
                        break
                    x = stage(x)
                    if stage_id >= len(self.stages) - (self.ret_stages-1):
                        out_dict[stage_id] = x
                
                return out_dict

        

def efficientvit_backbone_b0(**kwargs) -> LowFormerBackbone:
    backbone = LowFormerBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, LowFormerBackbone),
    )
    return backbone

# L3 channels!
# [64, 128, 256, 512, 1024]
#[1, 2, 2, 8, 8]
def lowformer_backbone_b1(**kwargs) -> LowFormerBackbone:
    # print(build_kwargs_from_config(kwargs, LowFormerBackbone))
    # multiply channels and depth!
    width_list = [16, 32, 64, 128, 256]
    depth_list = [1, 2, 3, 3, 4]
    if "fastit" in kwargs and kwargs["fastit"]:
        depth_list = [1, 2, 3, 5, 5] # TODO changed

    if "huge_model" in kwargs and kwargs["huge_model"]:
        depth_list[-2:] = [6,6]
        width_list =  [32, 64, 128, 256, 512]

    if "hugev2" in kwargs and kwargs["hugev2"]:
        depth_list = [1,1,2,6,6]
        width_list = [48,96,192, 384, 768]

    if "hugev3" in kwargs and kwargs["hugev3"]:
        width_list = [64,128,256,512,1024]
        depth_list = [3,4,4,8,8]

    if "newfastit" in kwargs and kwargs["newfastit"]:
        depth_list = [1,1,2,5,5]
    

    if "fastitv2" in kwargs and kwargs["fastitv2"]: #
        depth_list[0] = 0  
        depth_list[1] = 1
        depth_list[2] = 1
        depth_list[-2] += 2 
        depth_list[-1] -= 2 
        
    if "fastitv3" in kwargs and kwargs["fastitv3"]: #
        depth_list[0] = 1 # TODO
        depth_list[1] = 2 # TODO
        depth_list[2] = 2 # TODO
        ## rechange from fastitv2 
        depth_list[-2] -= 1
        depth_list[-1] += 1
        
        depth_list[-2] -= 1
        depth_list[-1] -= 1
    
    if "fastitv4" in kwargs and kwargs["fastitv4"]:
        depth_list[0] = 0 # TODO
        depth_list[1] = 1 # TODO
        

        # to fastit back
        depth_list[-2] -= 1
        depth_list[-1] += 1
        # increase depth later
        depth_list[-2] += 3
        depth_list[-1] += 1
        width_list = [20,32,64,112,216]
    

    
    if "model_mult" in kwargs:
        model_mult = kwargs["model_mult"]
        if "old_way_norm" in kwargs:
            width_list, depth_list = [int(i*model_mult) for i in width_list], [int(i*model_mult) for i in depth_list]
        else:    
            width_list, depth_list = [int(i*model_mult) for i in width_list], [round(i*model_mult) for i in depth_list]
        # width_list[-2], width_list[-1] = 8*(width_list[-2]//8), 8*(width_list[-1]//8)

        # print(width_list, depth_list)
    else:
        model_mult = 1

    if "smallit" in kwargs and kwargs["smallit"]:
        depth_list[:3] = [0,1,2]
        # depth_list[3] = 5
        width_list[0] = max(12, width_list[0])
        kwargs["head_dim_mul"] = True
        # kwargs["fuse_conv_all"] = True
    if "nohdimmul" in kwargs and kwargs["nohdimmul"]:
        kwargs["head_dim_mul"] = False
    
    if "bigit" in kwargs and kwargs["bigit"]:
        pass
        # width_list = [28,45,75,140,270]
        # depth_list[-2] += 1
    if "smallv2" in kwargs and kwargs["smallv2"]:
        depth_list[:3] = [0,1,1]
    
    if "smallv3" in kwargs and kwargs["smallv3"]:
        depth_list[-2:] = [3,4]
    
    if "middlev1" in kwargs and kwargs["middlev1"]:
        width_list =  [24, 48, 96, 192, 384]
        depth_list[-2:] = [6,6]

    if "middlev2" in kwargs and kwargs["middlev2"]:
        width_list =  [20, 40, 80, 160, 320]
        # depth_list[-2:] = [5,5]

    if "noattention" in kwargs and kwargs["noattention"]:
        depth_list = [1,2,2,7,7]
    
    if "removeatt" in kwargs and kwargs["removeatt"]:
        kwargs["noattention"] = True
    
    if "shallowit" in kwargs and kwargs["shallowit"]:
        depth_list[-2:] = [depth_list[-2]-2, depth_list[-1]-2]


    if "less_layers" in kwargs and kwargs["less_layers"]>0:
        depth_list = depth_list[:-kwargs["less_layers"]]

    backbone = LowFormerBackbone(
        width_list=width_list,
        depth_list=depth_list,
        dim=int(16*model_mult),
        **build_kwargs_from_config(kwargs, LowFormerBackbone),
    )
    return backbone, width_list


def efficientvit_backbone_b2(**kwargs) -> LowFormerBackbone:
    backbone = LowFormerBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        **build_kwargs_from_config(kwargs, LowFormerBackbone),
    )
    return backbone


def efficientvit_backbone_b3(**kwargs) -> LowFormerBackbone:
    backbone = LowFormerBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, LowFormerBackbone),
    )
    return backbone


class EfficientViTLargeBackbone(nn.Module):
    def __init__(
        self,
        width_list: List[int],
        depth_list: List[int],
        block_list: List[str] or None = None,
        expand_list: List[float] or None = None,
        fewer_norm_list: List[bool] or None = None,
        in_channels=3,
        qkv_dim=32,
        norm="bn2d",
        act_func="gelu",
    ) -> None:
        super().__init__()
        block_list = block_list or ["res", "fmb", "fmb", "mb", "att"]
        expand_list = expand_list or [1, 4, 4, 4, 6]
        fewer_norm_list = fewer_norm_list or [False, False, False, True, True]

        self.width_list = []
        self.stages = []
        # stage 0
        stage0 = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                block=block_list[0],
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=expand_list[0],
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[0],
            )
            stage0.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.stages.append(OpSequential(stage0))
        self.width_list.append(in_channels)

        for stage_id, (w, d) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            stage = []
            block = self.build_local_block(
                block="mb" if block_list[stage_id] not in ["mb", "fmb"] else block_list[stage_id],
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_list[stage_id] * 4,
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[stage_id],
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                if block_list[stage_id].startswith("att"):
                    stage.append(
                        LowFormerBlock(
                            in_channels=in_channels,
                            dim=qkv_dim,
                            expand_ratio=expand_list[stage_id],
                            scales=(3,) if block_list[stage_id] == "att@3" else (5,),
                            norm=norm,
                            act_func=act_func,
                        )
                    )
                else:
                    block = self.build_local_block(
                        block=block_list[stage_id],
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=1,
                        expand_ratio=expand_list[stage_id],
                        norm=norm,
                        act_func=act_func,
                        fewer_norm=fewer_norm_list[stage_id],
                    )
                    block = ResidualBlock(block, IdentityLayer())
                    stage.append(block)
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        block: str,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if block == "res":
            block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "fmb":
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "mb":
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        else:
            raise ValueError(block)
        return block

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        output_dict = {"input": x}
        for stage_id, stage in enumerate(self.stages):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


def efficientvit_backbone_l0(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 4, 4],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l1(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 6, 6],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l2(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l3(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[64, 128, 256, 512, 1024],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone

