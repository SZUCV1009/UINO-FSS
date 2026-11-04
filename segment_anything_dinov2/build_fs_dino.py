# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, uino_fss, TwoWayTransformer, vits


def remove_suffix(s, suffix):
    if s.endswith(suffix):
        return s[:-len(suffix)]
    return s

def build_imageEncoder(args):
    args.arch = remove_suffix(args.arch, "_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=args.img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if args.only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim

    if args.pretrained_weights:
        chkpt = torch.load(args.pretrained_weights)
        student.load_state_dict(chkpt, strict=True)
    return student, teacher, embed_dim


def build_promptEncoder(args):
    image_embedding_size = args.image_size // args.patch_size
    image_size = args.image_size
    prompt_encoder = PromptEncoder(
        embed_dim=args.prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )
    if args.pretrained_weights:
        chkpt = torch.load(args.pretrained_weights)
        prompt_encoder.load_state_dict(chkpt, strict=True)
    return prompt_encoder


def build_maskDecoder(args):
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=args.prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=args.prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    if args.pretrained_weights:
        chkpt = torch.load(args.pretrained_weights)
        mask_decoder.load_state_dict(chkpt, strict=True)
    return mask_decoder



def _build_fs_dino(
    cfg,
):
    imageEncoder, _, _ = build_imageEncoder(cfg.imageEncoder)
    promptEncoder = build_promptEncoder(cfg.promptEncoder)
    maskDecoder = build_maskDecoder(cfg.maskDecoder)

    sam=uino_fss(
        image_encoder=imageEncoder,
        prompt_encoder=promptEncoder,
        mask_decoder=maskDecoder,
        embed_dim=cfg.imageEncoder.embed_dim,
        out_chans=cfg.maskDecoder.prompt_embed_dim
    )
    return sam
