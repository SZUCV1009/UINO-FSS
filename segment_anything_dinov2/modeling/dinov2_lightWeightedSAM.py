from torch import nn
import torch.nn.functional as F
import math
import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Type, Optional
from .base.correlation import Correlation
from .learner import Patch4DMambaLearner
from .bottleneck_adapter import DistillAttentionModule, LayerNorm2d


class uino_fss(nn.Module):
    def __init__(
        self,
            image_encoder,
            prompt_encoder,
            mask_decoder,
            embed_dim,
            out_chans,
            input_size=(518,518),
            encoder_feat_size=37,
            depth=2,
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            use_rel_pos=False,
            rel_pos_zero_init=True,
            patch_size=14,
            mask_in_chans=16,
            distilled_channel=[256,128,256],
            outconv_in_channels=64,
            mask_conv_kernel_size=2,
            mask_conv_stride=2,
            up_layer_in_channels=64,
            up_layer_out_channels=128,
            pe_layer_size=256,
            stack_nums=[3, 6, 3],
    ) -> None:
        super(uino_fss, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.outc = OutConv(outconv_in_channels, 1)

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=mask_conv_kernel_size, stride=mask_conv_stride),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=mask_conv_kernel_size, stride=mask_conv_stride),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, out_chans, kernel_size=1),
        )

        self.apply(self.initialize_weights)

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.input_size = input_size
        self.encoder_feat_size = encoder_feat_size

        self.ba = DistillAttentionModule(
            distilled_channel, depth, num_heads, mlp_ratio, qkv_bias,
            norm_layer, act_layer, use_rel_pos, rel_pos_zero_init,
            input_size, patch_size, embed_dim, out_chans
        )

        self.pe_layer = PositionEmbeddingRandom(pe_layer_size // 2)
        self.pe_layer.positional_encoding_gaussian_matrix = nn.Parameter(self.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix.detach())
        self.alpha = torch.nn.Parameter(torch.tensor(0.0))

        self.stack_ids = torch.tensor([3, 9, 12], device=self.device)
        self.stack_nums = stack_nums
        self.mamba_learner = Patch4DMambaLearner(self.stack_nums)

        self.up1_n_n = Up_layer(up_layer_in_channels, up_layer_out_channels)
        self.up2_n_n = Up_layer(up_layer_out_channels, up_layer_in_channels)

        self.up1_light = Up_layer(1, 32)
        self.up2_light = Up_layer(32, 64)

        # self.freeze_modules()

        ###################################################################################################

        # self.mlp = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.GELU(),
        #     nn.Linear(512, 256)
        # )

    # def freeze_modules(self):
    #     for param in self.image_encoder.parameters():
    #         param.requires_grad = False
    #     for param in self.prompt_encoder.parameters():
    #         param.requires_grad = False
    #     for param in self.mask_decoder.parameters():
    #         param.requires_grad = True
    #     for param in self.ba.parameters():
    #         param.requires_grad = False
    #     self.pe_layer.positional_encoding_gaussian_matrix.requires_grad = False

    def freeze_modules(self, args):
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        if args.mask_decoder_train_flag:
            for param in self.mask_decoder.parameters():
                param.requires_grad = True
        else:
            for param in self.mask_decoder.parameters():
                param.requires_grad = False

        if args.ba_train_flag:
            for param in self.ba.parameters():
                param.requires_grad = True
        else:
            for param in self.ba.parameters():
                param.requires_grad = False

        if args.pe_train_flag:
            self.pe_layer.positional_encoding_gaussian_matrix.requires_grad = True
        else:
            self.pe_layer.positional_encoding_gaussian_matrix.requires_grad = False


    def masked_cos_sim(self, input, s_input, s_mask, stop_layer=4):
        self.image_encoder.eval()
        s_dict = self.image_encoder.forward_features(s_input.squeeze(1))
        support_feats = s_dict["x_norm_patchtokens"]
        self.support_layers_l = s_dict["list_all_layers"]

        t_dict = self.image_encoder.forward_features(input, stop_layer)
        target_feats = t_dict["x_norm_patchtokens"]
        self.target_layers_l = t_dict["list_all_layers"]
        self.early_layer_norm = t_dict["x_early_layer"]

        mask_resized = F.interpolate(s_mask.float(), size=(self.encoder_feat_size, self.encoder_feat_size), mode='nearest').squeeze(1)
        mask_resized = mask_resized.view(support_feats.shape[0], -1)
        mask_resized = mask_resized.unsqueeze(-1)
        mask_resized = torch.where(mask_resized==1, 1, 0)
        support_feats = support_feats * mask_resized

        cos_sim_map = torch.bmm(support_feats, target_feats.transpose(1, 2))
        cos_sim_map = cos_sim_map.mean(1)
        x_min = cos_sim_map.min(dim=1, keepdim=True)[0]
        x_max = cos_sim_map.max(dim=1, keepdim=True)[0]

        cos_sim_map = (cos_sim_map - x_min) / (x_max - x_min + 1e-6)
        cos_sim_map = cos_sim_map.view(support_feats.shape[0], int(cos_sim_map.shape[1] ** 0.5), int(cos_sim_map.shape[1] ** 0.5))

        return cos_sim_map, target_feats, support_feats

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='nearest')
            features[idx] = features[idx] * mask
        return features

    def get_dense_emb(self, s_mask, cos_sim_map, ce_flag=True, is_pcs=True):
        self.image_encoder.eval()
        s_mask_fg = torch.where(s_mask == 1, 1, 0)
        with torch.no_grad():
            query_feature_list = self.target_layers_l
            support_feature_list = self.support_layers_l.copy()
            support_feature_list = self.mask_feature(support_feature_list, s_mask_fg.squeeze(1).clone())
            corr_fg = Correlation.multilayer_correlation(query_feature_list, support_feature_list, self.stack_ids)

        s_mask_bg = torch.where(s_mask == 0, 1, 0)
        with torch.no_grad():
            query_feature_list = self.target_layers_l
            support_feature_list =  self.support_layers_l.copy()
            support_feature_list = self.mask_feature(support_feature_list, s_mask_bg.squeeze(1).clone())
            corr_bg = Correlation.multilayer_correlation(query_feature_list, support_feature_list, self.stack_ids)

        if ce_flag:
            corr_fg[0] = corr_fg[0] - corr_bg[0]
        cos_sim_map[torch.isnan(cos_sim_map)] = 0.0
        merge_map = torch.sigmoid(cos_sim_map.unsqueeze(1))
        alpha_ = torch.sigmoid(self.alpha)
        ts_map = self.mamba_learner(corr_fg)
        if is_pcs:
            feature = alpha_ * merge_map + (1 - alpha_) * ts_map
        else:
            feature = ts_map
        feature = self.up1_n_n(feature)
        merge_map_1 = F.interpolate(merge_map, size=(merge_map.shape[2] * 2, merge_map.shape[3] * 2), mode='bilinear',
                                    align_corners=False)
        if is_pcs: feature = feature + merge_map_1
        feature = self.up2_n_n(feature)
        merge_map_2 = F.interpolate(merge_map, size=(merge_map_1.shape[2] * 2, merge_map_1.shape[3] * 2),
                                    mode='bilinear',
                                    align_corners=False)
        if is_pcs: feature = feature + merge_map_2
        logits = self.outc(feature)

        dense_embedding = self.mask_downscaling(logits)
        logits = F.interpolate(logits, (518, 518), mode="bilinear", align_corners=True)

        return logits, dense_embedding

    def get_dense_onlycs(self, batch_cosine_map):
        if torch.isnan(batch_cosine_map).any():
            print("batch_cosine_map.nan")

        batch_cosine_map[torch.isnan(batch_cosine_map)] = 0.0
        merge_map = torch.sigmoid(batch_cosine_map.unsqueeze(1))

        feature = merge_map
        feature = self.up1_light(feature)
        merge_map_1 = F.interpolate(merge_map, size=(merge_map.shape[2]*2, merge_map.shape[3]*2), mode='bilinear', align_corners=False)
        feature = feature + merge_map_1
        feature = self.up2_light(feature)
        merge_map_2 = F.interpolate(merge_map, size=(merge_map_1.shape[2] * 2, merge_map_1.shape[3] * 2), mode='bilinear',
                                    align_corners=False)
        feature = feature + merge_map_2
        logits = self.outc(feature)

        dense_embedding = self.mask_downscaling(logits)

        logits = F.interpolate(logits, (518, 518), mode="bilinear", align_corners=True)
        return logits, dense_embedding

    def get_sparse_emb(self, s_feat):
        support_feats = s_feat.permute(0, 2, 1).view(s_feat.shape[0], s_feat.shape[2], self.encoder_feat_size, self.encoder_feat_size)

        support_emb = self.ba.neck(support_feats)
        support_emb = support_emb.view(s_feat.shape[0], 256, -1)
        support_proto = support_emb.mean(dim=2)

        sparse_embedding = support_proto.unsqueeze(1)
        sparse_embedding = torch.sin(sparse_embedding)

        return sparse_embedding

    def get_img_emb(self, x):
        B, L, C = x.shape
        x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), -1)
        x = self.ba.neck(x.permute(0, 3, 1, 2))
        x = x.permute(0,2,3,1)
        x = self.ba.distill_down_channel(x.permute(0, 3, 1, 2))
        x = x.permute(0, 2, 3, 1)
        for i in range(len(self.ba.attn_blocks_light_128)):
            x = self.ba.attn_blocks_light_128[i](x)
        image_embeddings = self.ba.distill_up_channel(x.permute(0, 3, 1, 2))
        return image_embeddings

    # def get_img_emb_only_mlp(self, x):
    #     B, L, C = x.shape
    #     x = self.mlp(x)   # [B, L 256]
    #     x = x.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), 256)
    #     x = x.permute(0, 3, 1, 2)
    #     return x

    def get_img_emb_only_mlp(self, x):
        B, L, C = x.shape
        x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), -1)
        x = self.ba.neck(x.permute(0, 3, 1, 2))
        return x


    def ba_forward(self, input_images, stop_layer=4):
        x = self.image_encoder.forward_features_previous_layer(input_images, stop_layer)
        B, L, C = x.shape
        x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), -1)
        x = self.ba.neck(x.permute(0, 3, 1, 2))
        x = x.permute(0,2,3,1)
        x = self.ba.distill_down_channel(x.permute(0, 3, 1, 2))
        x = x.permute(0, 2, 3, 1)
        for i in range(len(self.ba.attn_blocks_light_128)):
            x = self.ba.attn_blocks_light_128[i](x)
        image_embeddings = self.ba.distill_up_channel(x.permute(0, 3, 1, 2))
        return image_embeddings

    def forward(
        self,
        input,
        s_input=None,
        s_mask=None,
        stop_layer=4,
        is_light=False,
        is_sparse=True,
        ce_flag=True,
        is_ba=True,
        is_pcs=True,
        distill_only=False,
    ):
        if distill_only:
            return self.ba_forward(input, stop_layer)
        with torch.no_grad():
            batch_cosine_map, t_feat, s_feat = self.masked_cos_sim(input, s_input, s_mask, stop_layer)
        batch_cosine_map, t_feat, s_feat = batch_cosine_map.detach(), t_feat.detach(), s_feat.detach()
        if is_light:
            logits, dense_embedding = self.get_dense_onlycs(batch_cosine_map)
        else:
            logits, dense_embedding = self.get_dense_emb(s_mask, batch_cosine_map, ce_flag, is_pcs)
        with torch.no_grad():
            if is_ba:
                image_embeddings = self.get_img_emb(self.early_layer_norm)
            else:
                image_embeddings = self.get_img_emb_only_mlp(self.early_layer_norm)
        if is_sparse:
            sparse_embedding = self.get_sparse_emb(s_feat)
        else:
            sparse_embedding, _ = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

        image_embedding_size = (image_embeddings.shape[2], image_embeddings.shape[3])
        image_position_encoding = self.pe_layer(image_embedding_size).unsqueeze(0)

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_position_encoding,
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            multimask_output=False,
        )
        masks = F.interpolate(
            low_res_masks,
            size=(input.shape[2], input.shape[3]),
            mode="bilinear",
            align_corners=True,
        )

        return logits, masks

    def initialize_weights(self, module):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.positional_encoding_gaussian_matrix = nn.Parameter(
            scale * torch.randn((2, num_pos_feats))
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix.to(torch.float32)
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float32))  # B x N x C

class Up_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.up_conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)
