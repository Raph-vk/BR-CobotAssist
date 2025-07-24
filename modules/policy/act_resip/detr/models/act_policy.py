# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
ACT Policy implementation for transformer-based action prediction.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import build_ACT_model


def get_args_parser():
    """Get argument parser for ACT model configuration."""
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--batch_size', default=2, type=int) # not used
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int) # not used
    parser.add_argument('--lr_drop', default=200, type=int) # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_all', action='store_true')
    parser.add_argument('--train_params', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=False)
    parser.add_argument('--policy_class', default='ACT', action='store', type=str, help='policy_class, capitalize', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=False)
    parser.add_argument('--seed', default=0, action='store', type=int, help='seed', required=False)
    parser.add_argument('--num_epochs', default=3000, action='store', type=int, help='num_epochs', required=False)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--rollouts', action='store', type=int, help='rollouts', required=False)
    parser.add_argument('--model_name', action='store', type=str, help='model_name', required=False)

    return parser


def build_ACT_model_and_optimizer(args_override):
    """Build ACT model and optimizer with given configuration overrides."""
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        print(f"Overriding {k} with {v}")
        setattr(args, k, v)

    model = build_ACT_model(args)
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


class ACTPolicy(nn.Module):
    """
    ACT (Action Chunking with Transformers) policy model.
    
    This class wraps the ACT model with training and inference logic,
    including KL divergence loss computation and weighted action prediction.
    """
    
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # image = normalize(image)
        
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()

            loss_dict['kl'] = total_kld[0]           
            alpha = 0.5  # Weight factor for L1 loss; (1 - alpha) will be the weight for MSE loss

            # Compute per-element losses
            l1_loss = F.l1_loss(actions, a_hat, reduction='none')
            mse_loss = F.mse_loss(actions, a_hat, reduction='none')

            # Create a linearly decreasing weight vector
            seq_len = actions.size(1)
            weights = torch.linspace(1.0, 0.05, steps=seq_len).to(actions.device)
            weights = weights.view(1, seq_len, 1)  # Shape [1, seq_len, 1]

            # Apply weights and mask
            mask = (~is_pad).unsqueeze(-1).float()  # Shape [batch_size, seq_len, 1]
            combined_loss = alpha * l1_loss + (1 - alpha) * mse_loss
            weighted_loss = combined_loss * weights * mask

            # Compute the total weighted loss and normalize
            total_weight = (weights * mask).sum()
            loss = weighted_loss.sum() / total_weight

            # Update loss dictionary
            loss_dict['weighted_loss'] = loss
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss + loss_dict['kl'] * self.kl_weight

            return loss_dict

        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior

            print('qpos', qpos)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    """
    Compute KL divergence for VAE loss.
    
    Args:
        mu: Mean of the latent distribution
        logvar: Log variance of the latent distribution
        
    Returns:
        total_kld: Total KL divergence
        dimension_wise_kld: KL divergence per dimension
        mean_kld: Mean KL divergence
    """
    batch_size = mu.size(0)
    assert batch_size != 0
    
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
