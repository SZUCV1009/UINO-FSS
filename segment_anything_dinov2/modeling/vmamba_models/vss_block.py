
import torch
import torch.nn as nn
from .vmamba import VSSBlock
from .vmamba import SS2D
from collections import OrderedDict
from .vmamba import PatchMerging2D
from .vmamba import Permute
import torch.nn.functional as F
from einops import rearrange
import math

class CPConv4d(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, vssl, bias=True):
        super(CPConv4d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                               bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

        ################################################################################################
        drop_path = (0.1,) * vssl
        # self.vss1 = vmamba_4d(
        #         dim=out_channels,
        #         drop_path=drop_path,
        #         use_checkpoint=False,
        #         norm_layer=nn.LayerNorm,
        #         downsample=nn.Identity(),
        #         channel_first=False,
        #         # ===========================
        #         ssm_d_state=16,
        #         ssm_ratio=2.0,
        #         ssm_dt_rank="auto",
        #         ssm_act_layer=nn.SiLU,
        #         ssm_conv=3,
        #         ssm_conv_bias=True,
        #         ssm_drop_rate=0.0,
        #         ssm_init="v0",
        #         forward_type="v0",
        #         # ===========================
        #         mlp_ratio=4.0,
        #         mlp_act_layer=nn.GELU,
        #         mlp_drop_rate=0.0,
        #         gmlp=False,
        #         # ===========================
        #         _SS2D=SS2D
        #     )
        #
        # self.vss2 = vmamba_4d(
        #     dim=out_channels,
        #     drop_path=drop_path,
        #     use_checkpoint=False,
        #     norm_layer=nn.LayerNorm,
        #     downsample=nn.Identity(),
        #     channel_first=False,
        #     # ===========================
        #     ssm_d_state=16,
        #     ssm_ratio=2.0,
        #     ssm_dt_rank="auto",
        #     ssm_act_layer=nn.SiLU,
        #     ssm_conv=3,
        #     ssm_conv_bias=True,
        #     ssm_drop_rate=0.0,
        #     ssm_init="v0",
        #     forward_type="v0",
        #     # ===========================
        #     mlp_ratio=4.0,
        #     mlp_act_layer=nn.GELU,
        #     mlp_drop_rate=0.0,
        #     gmlp=False,
        #     # ===========================
        #     _SS2D=SS2D
        # )

        # self.gn = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        # self.ln = nn.LayerNorm(normalized_shape=out_channels)

        # self.vmamba_block = vmamba_4d(
        #     dim=out_channels,
        #     drop_path=drop_path,
        #     use_checkpoint=False,
        #     norm_layer=nn.LayerNorm,
        #     downsample=nn.Identity(),
        #     channel_first=False,
        #     # ===========================
        #     ssm_d_state=16,
        #     ssm_ratio=2.0,
        #     ssm_dt_rank="auto",
        #     ssm_act_layer=nn.SiLU,
        #     ssm_conv=3,
        #     ssm_conv_bias=True,
        #     ssm_drop_rate=0.0,
        #     ssm_init="v0",
        #     forward_type="v0",
        #     # ===========================
        #     mlp_ratio=4.0,
        #     mlp_act_layer=nn.GELU,
        #     mlp_drop_rate=0.0,
        #     gmlp=False,
        #     # ===========================
        #     _SS2D=SS2D
        # )

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
            self.len_h = len(idxh)
            self.len_w = len(idxw)
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def pad_to_multiple(self, x, patch_size):
        # x: (B, C, Hqw, Wqw, Hsw, Wsw)
        B, C, Hqw, Wqw, Hsw, Wsw = x.shape

        def pad_len(dim, p):
            return (math.ceil(dim / p) * p) - dim

        pad_Hqw = pad_len(Hqw, patch_size)
        pad_Wqw = pad_len(Wqw, patch_size)
        pad_Hsw = pad_len(Hsw, patch_size)
        pad_Wsw = pad_len(Wsw, patch_size)

        # pad in the four spatial dims: use F.pad which pads last dims first
        # order for F.pad: (..., Wsw, Hsw, Wqw, Hqw): so we reverse
        x = F.pad(x, (0, pad_Wsw, 0, pad_Hsw, 0, pad_Wqw, 0, pad_Hqw))  # only spatial dims
        return x, (pad_Hqw, pad_Wqw, pad_Hsw, pad_Wsw)

    def extract_patches(self, x, patch_size):
        """
        输入 x: (B, C, Hqw, Wqw, Hsw, Wsw)
        输出 patches: (B, N_patches, patch_volume, C), 其中 patch_volume = patch_size**4
        以及 spatial layout sizes 用于重组
        """
        B, C, Hqw, Wqw, Hsw, Wsw = x.shape
        x, pads = self.pad_to_multiple(x, patch_size)
        Hqw_p, Wqw_p, Hsw_p, Wsw_p = x.shape[2:]

        H_q = Hqw_p // patch_size
        W_q = Wqw_p // patch_size
        H_s = Hsw_p // patch_size
        W_s = Wsw_p // patch_size

        # 切 patch: B, H_q, W_q, H_s, W_s, C, p, p, p, p
        x = rearrange(
            x,
            'B C (H_q p1) (W_q p2) (H_s p3) (W_s p4) -> B H_q W_q H_s W_s C p1 p2 p3 p4',
            p1=patch_size, p2=patch_size, p3=patch_size, p4=patch_size
        )
        # 合并 patch 内的体素为 token 序列：每个 patch 是 patch_size^4 个位置，每个位置有 C 维
        # 先变成 B, N_patches, patch_volume, C
        num_patches = H_q * W_q * H_s * W_s
        patch_volume = patch_size ** 4
        patches = rearrange(
            x,
            'B H_q W_q H_s W_s C p1 p2 p3 p4 -> B (H_q W_q H_s W_s) (p1 p2 p3 p4) C'
        )  # shape: B, N_patches, patch_volume, C

        return patches, (H_q, W_q, H_s, W_s), pads

    def combine_patches(self, patches, layout, pads, patch_size):
        """
        逆 extract_patches：
        patches: (B, N_patches, patch_volume, C)  # patch_volume = patch_size**4
        layout: (H_q, W_q, H_s, W_s)
        pads: (pad_Hqw, pad_Wqw, pad_Hsw, pad_Wsw)  # 来自 pad_to_multiple
        返回: x 恢复到原始未 pad 之前的形状 (B, C, Hqw, Wqw, Hsw, Wsw)
        """
        H_q, W_q, H_s, W_s = layout
        pad_Hqw, pad_Wqw, pad_Hsw, pad_Wsw = pads  # 这些是加了多少 pad 在末端
        B, N, patch_volume, C = patches.shape

        # 先把 (B, N_patches, patch_volume, C) 还原成
        # B, H_q, W_q, H_s, W_s, C, p1, p2, p3, p4
        x = rearrange(
            patches,
            'B (H_q W_q H_s W_s) (p1 p2 p3 p4) C -> B H_q W_q H_s W_s C p1 p2 p3 p4',
            H_q=H_q, W_q=W_q, H_s=H_s, W_s=W_s,
            p1=patch_size, p2=patch_size, p3=patch_size, p4=patch_size
        )
        # 把 patch 还原回空间维度（含 padding）
        x = rearrange(
            x,
            'B H_q W_q H_s W_s C p1 p2 p3 p4 -> B C (H_q p1) (W_q p2) (H_s p3) (W_s p4)'
        )  # 这是 pad 之后的完整体

        # 把 padding 去掉：每个 spatial 维度末端 slice 掉 pad 的部分
        def unpad(tensor, pad_hqw, pad_wqw, pad_hsw, pad_wsw):
            _, _, Hqw_p, Wqw_p, Hsw_p, Wsw_p = tensor.shape
            hqw_end = -pad_hqw if pad_hqw > 0 else None
            wqw_end = -pad_wqw if pad_wqw > 0 else None
            hsw_end = -pad_hsw if pad_hsw > 0 else None
            wsw_end = -pad_wsw if pad_wsw > 0 else None
            return tensor[
                   :,
                   :,
                   :hqw_end,
                   :wqw_end,
                   :hsw_end,
                   :wsw_end
                   ]

        x = unpad(x, pad_Hqw, pad_Wqw, pad_Hsw, pad_Wsw)
        return x  # shape (B, C, Hqw_orig, Wqw_orig, Hsw_orig, Wsw_orig)

    def forward(self, x):
        if self.stride[2:][-1] > 1:
            out1 = self.prune(x)
        else:
            out1 = x
        bsz, inch, ha, wa, hb, wb = out1.size()
        out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1 = self.conv1(out1)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, hb, wb, o_ha, o_wa, outch).permute(0, 5, 3, 4, 1, 2).contiguous()
        # residual = out1
        # out1_normed = self.gn(out1)
        # out1 = out1_normed.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, outch, o_ha, o_wa)
        # out1 = F.interpolate(out1, size=(7, 7), mode='bilinear', align_corners=False)
        # out1 = self.ln(self.vss1(out1)).permute(0,3,1,2)
        # out1 = F.interpolate(out1, size=(o_ha, o_wa), mode='bilinear', align_corners=False)
        # out1 = out1.reshape(bsz, hb, wb, o_ha, o_wa, outch).permute(0, 5, 3, 4, 1, 2).contiguous()
        # out1 = residual + out1


        bsz, inch, ha, wa, hb, wb = x.size()
        out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out2 = self.conv2(out2)
        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        # residual = out2
        # out2_normed = self.gn(out2)
        # out2 = out2_normed.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, outch, o_hb, o_wb)
        # out2 = F.interpolate(out2, size=(7, 7), mode='bilinear', align_corners=False)
        # out2 = self.ln(self.vss2(out2)).permute(0,3,1,2)
        # out2 = F.interpolate(out2, size=(o_hb, o_wb), mode='bilinear', align_corners=False)
        # out2 = out2.reshape(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        # out2 = residual + out2

        if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
            out2 = out2.squeeze()

        y = out1 + out2

        # patch_size = 4
        # patches, layout, pads = self.extract_patches(y, patch_size)
        # patches = patches.view(-1, patch_size * patch_size, patch_size * patch_size, y.shape[1])
        # patches_out = self.vmamba_block(patches.permute(0,3,1,2))
        # patches_out = patches_out.view(y.shape[0],layout[0]*layout[1]*layout[2]*layout[3],patch_size**4,y.shape[1]).contiguous()
        # y = self.combine_patches(patches_out, layout, pads, patch_size)

        return y

class vmamba_4d(nn.Module):
    def __init__(
            self,
            dim=16,
            drop_path=[0.1, 0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            channel_first=False,
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v0",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            # ===========================
            _SS2D="v0",
            **kwargs,
    ):
        super().__init__()

        depth = len(drop_path)
        blocks1 = []
        for d in range(depth):
            blocks1.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS2D=_SS2D,
            ))
        self.vssb = nn.Sequential(
            Permute(0, 2, 3, 1),
            *blocks1  # blocks1 是一个 nn.Module 列表
        )



    def forward(self, x):
        return self.vssb(x)
