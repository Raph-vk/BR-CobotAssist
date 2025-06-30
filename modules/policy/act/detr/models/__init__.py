# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae

def build_ACT_model(args):
    return build_vae(args)

# Lazy import to avoid circular dependency
def __getattr__(name):
    if name == 'ACTPolicy':
        from .act_policy import ACTPolicy
        return ACTPolicy
    elif name == 'kl_divergence':
        from .act_policy import kl_divergence
        return kl_divergence
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
