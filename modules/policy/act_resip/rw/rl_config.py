import os
import json
import torch
from typing import Tuple
from datetime import datetime
from dataclasses import dataclass
# Add project root to sys.path for ACT_utils import
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from .act_config import ACTConfig

@dataclass
class RLConfig:
	# Define your config attributes with default values
	algorithm: str = "RW-ACT+ResiP"
	wandb_mode: str = "offline"
	verbose_logging: bool = True

	episode_steps: int = 10
	update_episodes: int = 3
	update_epochs: int = 30

	num_cameras: int = 4
	reward_scaling_factor: float = 5								# Factor with which rewards are multiplied, important for v_loss
	truncation_as_done: bool = True
	normalize: bool = True
	normalize_clip: int = 2
	normalize_eps: float = 1e-16
	gripper_active_threshold:float = 0.5

	qpos_shape: Tuple[int] = (7,)
	vision_shape: Tuple[int, int, int] = (4, 424, 240)	# (num_cameras, channels, height, width)
	action_shape: int = (7,)
	flag_shape: Tuple[int] = (1,)
	random_shift_augmentation_padding: int = 6							# Use random shift augmentation method

	detectron_device: str = "cuda"
	detectron_model_name: str = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
	detectron_cfg = None
	detectron_model_path: str = "/home/tos-pc3/Desktop/tos_app/modules/policy/act_resip/rw/detectron_models/ball_gripper_platform2.pth"
	detectron_visualize: bool = False
	detectron_check_frames: int = 5
	confidence_threshold: float = 0.5	# To-do up with robot
	depth_min: float = 0.55
	depth_max: float = 0.7
	center_x_min: int = 415
	center_x_max: int = 440
	center_y_min: int = 236
	center_y_max: int = 250				# To-do reset to 246 with robot

	alpha: float = 0.1												# Residual action scaling factor
	actor_lr: float = 0.0003
	critic_lr: float = 0.005
	weight_decay: float = 1e-4
	actor_num_warmup_steps: int = 5
	critic_num_warmup_steps: int = 0
	optimizer_steps: int = 1000
	
	eps: float = 1e-5
	betas: float = (0.9, 0.999)
	discount_factor: float = 0.999
	gae_lambda: float = 0.95
	clip_v_loss: bool = False
	clip_coef: float = 0.2
	ent_coef: float = 0.0
	v_loss_factor: float = 1
	residual_l1_coef: float = 0
	residual_l2_coef: float = 0
	kl_threshold: float = 0.1

	actor_hidden_dim: int = 512
	actor_num_layers: int = 4
	actor_output_std: float = 0
	actor_bias_on_last_layer: bool = False
	critic_hidden_dim: int = 256
	critic_num_layers: int = 2
	critic_output_std: float = 0.25
	critic_bias_on_last_layer: bool = True
	critic_last_layer_biast_const: float = 0.25

	act: ACTConfig = None
	backbone: str = "resnet18"
	start_time: str = datetime.now().strftime("%d-%m %H-%M")
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	save_path: str = ""
	ckpt_path: str = ""
	wandb_run_name: str = ""

	# def __init__(self, act_cfg):
	# 	# act = act_cfg
	# 	print("")
		
	
	@classmethod
	def from_json(cls, file_path: str) -> "RLConfig":
		with open(os.path.join(file_path, "rl_config.json"), "r") as f:
			data = json.load(f)
		return cls(**data)
			
	def __repr__(self):
		fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
		return f"{self.__class__.__name__}({fields})"