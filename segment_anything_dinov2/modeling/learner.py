
import torch.nn as nn
import torch.nn.functional as F

from .base.conv4d import CenterPivotConv4d as Conv4d
from .vmamba_models.vss_block import vmamba_4d
from .vmamba_models.vss_block import CPConv4d as Cp_conv4d
from .vmamba_models.vmamba import SS2D
from einops import rearrange
import math


class Patch4DMambaLearner(nn.Module):
    def __init__(
        self,
        inch,
        outch1: int = 16,
        outch2: int = 32,
        outch3: int = 64,
        outch1_1: int = 16,
        outch2_1: int = 32,
        outch1_2: int = 32,
        outch2_2: int = 64,
        drop_path: tuple = (0.1,),
        dim1: int = 32,
        dim2: int = 64,
        use_checkpoint: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: nn.Module = nn.Identity(),
        channel_first: bool = False,
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_dt_rank: str = "auto",
        ssm_act_layer: nn.Module = nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias: bool = True,
        ssm_drop_rate: float = 0.0,
        ssm_init: str = "v0",
        forward_type: str = "v0",
        mlp_ratio: float = 4.0,
        mlp_act_layer: nn.Module = nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp: bool = False,
        _SS2D: nn.Module = SS2D
    ):
        super().__init__()

        self.encoder_layer4_1 = self.make_building_block(
            in_channel=inch[0],
            out_channels=[outch1_1, outch2_1],
            kernel_sizes=[3, 3],
            spt_strides=[2, 2]
        )
        self.encoder_layer4_2 = self.make_building_block(
            in_channel=outch2_1,
            out_channels=[outch1_2, outch2_2],
            kernel_sizes=[3, 3],
            spt_strides=[2, 2]
        )

        vmamba_kwargs = {
            "drop_path": drop_path,
            "use_checkpoint": use_checkpoint,
            "norm_layer": norm_layer,
            "downsample": downsample,
            "channel_first": channel_first,
            "ssm_d_state": ssm_d_state,
            "ssm_ratio": ssm_ratio,
            "ssm_dt_rank": ssm_dt_rank,
            "ssm_act_layer": ssm_act_layer,
            "ssm_conv": ssm_conv,
            "ssm_conv_bias": ssm_conv_bias,
            "ssm_drop_rate": ssm_drop_rate,
            "ssm_init": ssm_init,
            "forward_type": forward_type,
            "mlp_ratio": mlp_ratio,
            "mlp_act_layer": mlp_act_layer,
            "mlp_drop_rate": mlp_drop_rate,
            "gmlp": gmlp,
            "_SS2D": _SS2D
        }

        self.vmamba_block_1 = vmamba_4d(dim=dim1, **vmamba_kwargs)
        self.vmamba_block_2 = vmamba_4d(dim=dim2, **vmamba_kwargs)

        # self.encoder_layer4 = self.make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])

    def make_building_block(self, in_channel, out_channels, kernel_sizes, spt_strides, group=4):
        assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

        building_block_layers = []
        for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
            inch = in_channel if idx == 0 else out_channels[idx - 1]
            ksz4d = (ksz,) * 4
            str4d = (1, 1) + (stride,) * 2
            pad4d = (ksz // 2,) * 4

            building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
            building_block_layers.append(nn.GroupNorm(group, outch))
            building_block_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*building_block_layers)

    def pad_to_multiple(self, x, patch_size):
        """
        Pad the input tensor x to ensure the spatial dimensions (height, width, etc.) are
        divisible by the given patch size.

        Args:
            x: Tensor of shape (B, C, Hqw, Wqw, Hsw, Wsw)
                B: Batch size
                C: Number of channels
                Hqw, Wqw, Hsw, Wsw: Spatial dimensions (height, width) in different scales
            patch_size: The size of each patch (the target multiple for padding)

        Returns:
            x: Padded tensor of shape (B, C, Hqw_p, Wqw_p, Hsw_p, Wsw_p)
            pads: Tuple of padding values (pad_Hqw, pad_Wqw, pad_Hsw, pad_Wsw)
        """
        B, C, Hqw, Wqw, Hsw, Wsw = x.shape

        # Function to calculate how much padding is needed for each dimension
        def pad_len(dim, p):
            return (math.ceil(dim / p) * p) - dim

        # Calculate padding required for each spatial dimension
        pad_Hqw = pad_len(Hqw, patch_size)
        pad_Wqw = pad_len(Wqw, patch_size)
        pad_Hsw = pad_len(Hsw, patch_size)
        pad_Wsw = pad_len(Wsw, patch_size)

        # Apply padding to the spatial dimensions using F.pad
        # The order of padding for F.pad is (left, right, top, bottom) for each dimension
        # Since F.pad applies padding to the last dimensions first, we reverse the order here
        x = F.pad(x, (0, pad_Wsw, 0, pad_Hsw, 0, pad_Wqw, 0, pad_Hqw))  # Only spatial dims are padded

        # Return padded tensor and the corresponding padding values
        return x, (pad_Hqw, pad_Wqw, pad_Hsw, pad_Wsw)


    def extract_patches(self, x, patch_size):
        """
        Extract patches from the input x:
        - Input x: (B, C, Hqw, Wqw, Hsw, Wsw)
        - Output patches: (B, N_patches, patch_volume, C), where patch_volume = patch_size**4
        - Returns spatial layout information for reconstruction
        """
        B, C, Hqw, Wqw, Hsw, Wsw = x.shape

        # Apply padding through pad_to_multiple
        x, pads = self.pad_to_multiple(x, patch_size)
        Hqw_p, Wqw_p, Hsw_p, Wsw_p = x.shape[2:]

        # Calculate the number of patches for each spatial dimension
        H_q = Hqw_p // patch_size
        W_q = Wqw_p // patch_size
        H_s = Hsw_p // patch_size
        W_s = Wsw_p // patch_size

        # Step 1: Slice the patches: B, H_q, W_q, H_s, W_s, C, p, p, p, p
        x = rearrange(
            x,
            'B C (H_q p1) (W_q p2) (H_s p3) (W_s p4) -> B H_q W_q H_s W_s C p1 p2 p3 p4',
            p1=patch_size, p2=patch_size, p3=patch_size, p4=patch_size
        )

        # Step 2: Combine the voxel positions within each patch into a token sequence:
        # Each patch has patch_size^4 positions, and each position has C channels
        num_patches = H_q * W_q * H_s * W_s
        patch_volume = patch_size ** 4
        patches = rearrange(
            x,
            'B H_q W_q H_s W_s C p1 p2 p3 p4 -> B (H_q W_q H_s W_s) (p1 p2 p3 p4) C'
        )  # shape: B, N_patches, patch_volume, C

        return patches, (H_q, W_q, H_s, W_s), pads


    def combine_patches(self, patches, layout, pads, patch_size):
        """
        Inverse operation: reconstruct the original shape from the extracted patches
        - Input patches: (B, N_patches, patch_volume, C)  # patch_volume = patch_size**4
        - Input layout: (H_q, W_q, H_s, W_s)
        - Input pads: (pad_Hqw, pad_Wqw, pad_Hsw, pad_Wsw)  # From pad_to_multiple
        - Output: x reconstructed to the original shape before padding (B, C, Hqw, Wqw, Hsw, Wsw)
        """
        H_q, W_q, H_s, W_s = layout
        pad_Hqw, pad_Wqw, pad_Hsw, pad_Wsw = pads  # Padding information
        B, N, patch_volume, C = patches.shape

        # Step 1: Convert (B, N_patches, patch_volume, C) back to (B, H_q, W_q, H_s, W_s, C, p1, p2, p3, p4)
        x = rearrange(
            patches,
            'B (H_q W_q H_s W_s) (p1 p2 p3 p4) C -> B H_q W_q H_s W_s C p1 p2 p3 p4',
            H_q=H_q, W_q=W_q, H_s=H_s, W_s=W_s,
            p1=patch_size, p2=patch_size, p3=patch_size, p4=patch_size
        )

        # Step 2: Reconstruct the spatial dimensions (with padding)
        x = rearrange(
            x,
            'B H_q W_q H_s W_s C p1 p2 p3 p4 -> B C (H_q p1) (W_q p2) (H_s p3) (W_s p4)'
        )  # This is the padded version

        # Step 3: Remove padding by slicing the padding parts from each spatial dimension
        def unpad(tensor, pad_hqw, pad_wqw, pad_hsw, pad_wsw):
            _, _, Hqw_p, Wqw_p, Hsw_p, Wsw_p = tensor.shape

            # Slice each dimension according to the padding
            hqw_end = -pad_hqw if pad_hqw > 0 else None
            wqw_end = -pad_wqw if pad_wqw > 0 else None
            hsw_end = -pad_hsw if pad_hsw > 0 else None
            wsw_end = -pad_wsw if pad_wsw > 0 else None

            return tensor[
                :,  # Keep all batches and channels
                :,  # Keep all channels
                :hqw_end,  # Remove padding from the height dimension
                :wqw_end,  # Remove padding from the width dimension
                :hsw_end,  # Remove padding from the height dimension
                :wsw_end   # Remove padding from the width dimension
            ]

        # Perform the padding removal
        x = unpad(x, pad_Hqw, pad_Wqw, pad_Hsw, pad_Wsw)

        return x  # Return the reconstructed shape (B, C, Hqw_orig, Wqw_orig, Hsw_orig, Wsw_orig)

    # def forward(self, hypercorr_pyramid):

    #     # Encode hypercorrelations from each layer (Squeezing building blocks)
    #     hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
    #     bsz, ch, ha, wa, hb, wb = hypercorr_sqz4.size()


    #     hypercorr_encoded = hypercorr_sqz4.view(bsz, ch, ha, wa, -1).mean(dim=-1)
    #     hypercorr_encoded = F.interpolate(hypercorr_encoded, size=(37, 37), mode='bilinear')


    #     return hypercorr_encoded


    def forward(self, hypercorr_pyramid):
        # Encode hypercorrelations from each layer (Squeezing building blocks)
        b, c, h1, w1, h2, w2 = hypercorr_pyramid[0].shape
        hypercorr_sqz4 = self.encoder_layer4_1(hypercorr_pyramid[0])

        residual = hypercorr_sqz4
        patch_size = 4
        patches, layout, pads = self.extract_patches(hypercorr_sqz4, patch_size)
        patches = patches.view(-1, patch_size * patch_size, patch_size * patch_size, hypercorr_sqz4.shape[1])
        patches_out = self.vmamba_block_1(patches.permute(0, 3, 1, 2))
        patches_out = patches_out.view(hypercorr_sqz4.shape[0], layout[0] * layout[1] * layout[2] * layout[3], patch_size ** 4,
                                       hypercorr_sqz4.shape[1]).contiguous()
        hypercorr_sqz4 = self.combine_patches(patches_out, layout, pads, patch_size)
        hypercorr_sqz4 = hypercorr_sqz4 + residual

        hypercorr_sqz4 = self.encoder_layer4_2(hypercorr_sqz4)

        residual = hypercorr_sqz4
        patch_size = 4
        patches, layout, pads = self.extract_patches(hypercorr_sqz4, patch_size)
        patches = patches.view(-1, patch_size * patch_size, patch_size * patch_size, hypercorr_sqz4.shape[1])
        patches_out = self.vmamba_block_2(patches.permute(0, 3, 1, 2))
        patches_out = patches_out.view(hypercorr_sqz4.shape[0], layout[0] * layout[1] * layout[2] * layout[3],
                                       patch_size ** 4,
                                       hypercorr_sqz4.shape[1]).contiguous()
        hypercorr_sqz4 = self.combine_patches(patches_out, layout, pads, patch_size)
        hypercorr_sqz4 = hypercorr_sqz4 + residual

        bsz, ch, ha, wa, hb, wb = hypercorr_sqz4.size()
        hypercorr_encoded = hypercorr_sqz4.view(bsz, ch, ha, wa, -1).mean(dim=-1)
        hypercorr_encoded = F.interpolate(hypercorr_encoded, size=(h1, w1), mode='bilinear')


        return hypercorr_encoded
