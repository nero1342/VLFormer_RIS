# Copyright (c) Facebook, Inc. and its affiliates.
from .vlformer_decoder import VideoMultiScaleVLformerDecoder
from .vlformer_decoder import TRANSFORMER_DECODER_REGISTRY

def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)

