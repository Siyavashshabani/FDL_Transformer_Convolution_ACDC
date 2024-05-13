## fusion block
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.utils.deprecate_utils import deprecated_arg

rearrange, _ = optional_import("einops", name="rearrange")


from architecture.transformer import transformerBranch
from architecture.ConvEncoder import ConvBranch 
from architecture.fusion import BiFusion_block_3d
from utils.utils import UnetrUpBlockLastLayer



class TransformerConv(nn.Module):

    patch_size: Final[int] = 2

    @deprecated_arg(
        name="img_size",
        since="1.3",
        removed="1.5",
        msg_suffix="The img_size argument is not required anymore and "
        "checks on the input size are run during forward().",
    )
    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ) -> None:
        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        self.normalize = normalize

        self.TransformerBranch = transformerBranch(
        img_size=(96, 96, 96),
        in_channels=1,
        feature_size=48,
        use_checkpoint=True,
        )
        self.ConvolutionalBranch = ConvBranch()

        self.fusion0 = BiFusion_block_3d(ch_1=48, ch_2=48, r_2=2, ch_int=48, ch_out=48, drop_rate=0.2)
        self.fusion1 = BiFusion_block_3d(ch_1=96, ch_2=96, r_2=2, ch_int=96, ch_out=96, drop_rate=0.2)
        self.fusion2 = BiFusion_block_3d(ch_1=192, ch_2=192, r_2=2, ch_int=192, ch_out=192, drop_rate=0.2)
        self.fusion3 = BiFusion_block_3d(ch_1=384, ch_2=384, r_2=2, ch_int=384, ch_out=384, drop_rate=0.2)
        self.fusion4 = BiFusion_block_3d(ch_1=768, ch_2=768, r_2=2, ch_int=768, ch_out=768, drop_rate=0.2)




        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels= 4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder0 = UnetrUpBlockLastLayer(
            spatial_dims=spatial_dims,
            in_channels=feature_size ,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        
    def forward(self, x_in):
        ## transformer branch 
        transformer_outputs = self.TransformerBranch(x_in)
        # print("output of transoformer block:")
        # print(transformer_outputs[0].shape, transformer_outputs[1].shape, transformer_outputs[2].shape, transformer_outputs[3].shape)
        ## convolutional branch 
        onvolutional_outputs = self.ConvolutionalBranch(x_in)
        # print("output of convolutional block:")
        # print(onvolutional_outputs[0].shape, onvolutional_outputs[1].shape, onvolutional_outputs[2].shape, onvolutional_outputs[3].shape, onvolutional_outputs[4].shape)
        
        ##fusion blocks  
        f0 = self.fusion0(onvolutional_outputs[1], transformer_outputs[0])
        f1 = self.fusion1(onvolutional_outputs[2], transformer_outputs[1])
        f2 = self.fusion2(onvolutional_outputs[3], transformer_outputs[2])
        f3 = self.fusion3(onvolutional_outputs[4], transformer_outputs[3])
        # f4 = self.fusion4(onvolutional_outputs[4], transformer_outputs[4])

        ## deconvolutional branch 

        dec3 = self.decoder3(f3, f2)

        # print("dec3: ",dec3.shape, "f1: ",f1.shape)
        dec2 = self.decoder2(dec3, f1 )

        dec1 = self.decoder1(dec2, f0)
        # print("dec1: ",dec1.shape, "onvolutional_outputs[1]: ",onvolutional_outputs[1].shape)
        dec0 = self.decoder0(dec1, onvolutional_outputs[1])

        # print("shape of before last layer", dec0.shape)
        logits = self.out(dec0)
        
        return logits
    
    

