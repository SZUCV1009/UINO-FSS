import os
import torch
import torch.nn as nn
from typing import Type, Optional, Tuple

from segment_anything import sam_model_registry
from pathlib import Path

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent.parent.parent
path = os.path.join(current_file_path.parent.parent.parent,"checkpoints/sam_vit_b_01ec64.pth")
sam = sam_model_registry["vit_b"](checkpoint=path)

class DistillAttentionModule(nn.Module):
    def __init__(self, distilled_channel, depth, num_heads, mlp_ratio, qkv_bias,
                 norm_layer, act_layer, use_rel_pos, rel_pos_zero_init,
                 input_size, patch_size, embed_dim, out_chans):
        super().__init__()

        # Distill Down Channel
        self.distill_down_channel = nn.Sequential(
            nn.Conv2d(
                distilled_channel[0],
                distilled_channel[1],
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(distilled_channel[1]),
            nn.GELU(),
        )

        # Distill Up Channel
        self.distill_up_channel = nn.Sequential(
            nn.Conv2d(
                distilled_channel[1],
                distilled_channel[2],
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(distilled_channel[2]),
            nn.GELU(),
        )

        # Attention Blocks
        self.attn_blocks_light_128 = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=distilled_channel[1],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                input_size=(input_size[0] // patch_size, input_size[0] // patch_size),
            )
            self.attn_blocks_light_128.append(block)

        # Neck Module
        # self.neck = nn.Sequential(
        #     nn.Conv2d(
        #         embed_dim,
        #         out_chans,
        #         kernel_size=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(out_chans),
        #     nn.Conv2d(
        #         out_chans,
        #         out_chans,
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(out_chans),
        # )
        self.neck = sam.image_encoder.neck

    def forward(self, x):
        x = self.distill_down_channel(x)
        for block in self.attn_blocks_light_128:
            x = block(x)
        x = self.distill_up_channel(x)
        x = self.neck(x)

        return x

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
