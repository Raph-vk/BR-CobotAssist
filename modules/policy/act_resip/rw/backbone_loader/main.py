# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model

import IPython
e = IPython.embed

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-5, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--batch_size', default=2, type=int) # not used
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int) # not used
    parser.add_argument('--lr_drop', default=200, type=int) # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet34', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', default=False, action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # parser.add_argument('--camera_names', default=["cam_1", "cam_2"], type=list, # will be overridden
    #                     help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=9, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=1024, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=32, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=75, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', default=False, action='store_true')

    # * Segmentation
    parser.add_argument('--masks', default=False, action='store_true',
                        help="Train segmentation head if the flag is provided")

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_all', default=False, action='store_true')
    parser.add_argument('--train_params', default=False, action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=False)
    parser.add_argument('--policy_class', default='ACT', action='store', type=str, help='policy_class, capitalize', required=False)
    parser.add_argument('--task_name', default="robot_training", action='store', type=str, help='task_name', required=False)
    parser.add_argument('--seed', default=0, action='store', type=int, help='seed', required=False)
    parser.add_argument('--num_epochs', default=3000, action='store', type=int, help='num_epochs', required=False)
    parser.add_argument('--kl_weight', default=100, action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', default=75, action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', default=False, action='store_true')
    parser.add_argument('--rollouts', default=None, action='store', type=int, help='rollouts', required=False)
    parser.add_argument('--model_name', default=None, action='store', type=str, help='model_name', required=False)

    return parser


def build_ACT_model_and_optimizer(args_override, cfg=None):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args(args=[])  # Don't parse actual command line arguments

    # for k, v in args_override.items():
    #     print(f"Overriding {k} with {v}")
    #     setattr(args, k, v)

    model = build_ACT_model(args, cfg)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

# args = get_args_parser()
# build_ACT_model_and_optimizer(args)
