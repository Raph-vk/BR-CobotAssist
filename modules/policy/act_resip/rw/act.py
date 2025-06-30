import os, sys
import torch
# Add project root to sys.path for ACT_utils import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ACT_utils.ACT import ACTPolicy, build_backbone
from ACT_utils.config import Config

class ACT():
    """
    Behaviour‑cloning model that produces chunked actions and never trains online (frozen weights).
    """

    def __init__(self, ckpt_dir: str):
        config = Config()
        # pathi = "/home/tos-pc3/Desktop/TOS-ResiP/act_models/act_models_peach-sponge-657_26-05 16-56/best_loss.pth"
        # state_sim = torch.load(pathi ,map_location="cpu")
        # state_keys_sim = state_sim["model_state_dict"].keys()

        pathi = os.path.join(ckpt_dir, "policy_best_epoch_4538.ckpt")
        state_rw = torch.load(pathi ,map_location="cpu")
        state_keys_rw = state_rw.keys()
        
        def extract_backbone_state_dict(checkpoint, backbone_idx):
            prefix = f"model.backbones.{backbone_idx}.0.body."
            state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith(prefix):
                    new_key = k[len(prefix):]  # Strip the prefix
                    state_dict[new_key] = v
                    # print(k, len(v))
            return state_dict

        # Build and load 4 backbones
        backbones = []
        for i in range(4):
            # 1. Build the backbone using your function (pass config as needed)
            backbone = build_backbone(config)
            # 2. Extract the state_dict for this backbone
            state_dict = extract_backbone_state_dict(state_rw, i)
            # 3. Load the state_dict into the .body of the backbone
            backbone[0].body.load_state_dict(state_dict, strict=False)
            backbones.append(backbone)

        print(backbone[0].body.conv1.in_channels)

        import numpy as np
        import matplotlib.pyplot as plt

        # Create the array as before
        x = torch.randn(1, 4, 240, 240)  # (batch, channels, height, width)
        output = backbone[0](x)
        print(output["0"].shape)


    @torch.no_grad()
    def _predict_next_actions(self, qpos, vision_data) -> torch.tensor:
        qpos, env_state, vision = self._split_current_obs()
        action_chunk: torch.tensor  = self.model(qpos, None, vision_data)[0]
        return action_chunk
    

ACT("/home/tos-pc3/Desktop/TOS-ResiP/act_models/real-world")