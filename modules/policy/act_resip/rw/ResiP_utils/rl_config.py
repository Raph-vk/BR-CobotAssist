import os
import json
from typing import List, Optional, Tuple, TypedDict
from datetime import datetime
from dataclasses import dataclass, field

from ACT_utils.config import Config

class PlotInfo(TypedDict):
    start_episode: int
    start_step:   int
    end_episode:   Optional[int]
    end_step:      Optional[int]

@dataclass
class RLConfig:
	# Define your config attributes with default values
	algorithm: str = "ACT+ResiP"
	base_policy_path: str = "resip_models/act 11 visual"					# Relative path to base model
	wandb_mode: str = "offline"

	save_iteration_threshold: int = 3
	save_iteration_interval: int = 5
	env_success_reward: int = 3000									# Reward required for evaluation to be considered a success
	base_policy_best_reward: int = -1								# Best reward of the base policy
	reward_scaling_factor: float = 5								# Factor with which rewards are multiplied, important for v_loss
	train_sparse_rewards: bool = True
	rollout_chunk: bool = True									# Use action chunking rollouts for evaluation
	eval_rollout_chunk: bool = True
	truncation_as_done: bool = True
	normalize: bool = True
	normalize_clip: int = 2
	normalize_eps: float = 1e-16

	# Consecutive succes metric hyperparameters
	consecutive_reward_threshold: int = 30								# Threshold from which best reward will be saved
	consecutive_steps_reward: int = 10

	alpha: float = 0.1												# Residual action scaling factor
	actor_lr: float = 0.0003
	critic_lr: float = 0.005
	weight_decay: float = 1e-4
	actor_num_warmup_steps: int = 5
	critic_num_warmup_steps: int = 0
	
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

	training_steps: int = 10e9
	steps_per_simulation: int = 200							
	update_epochs: int = 30
	num_envs: int = 50
	envs_multiple: int = 10

	eval_trials: int = 100											# Trials used for evaluation
	eval_interval: int = 1										# Iterations between evaluating
	eval_save_videos: int = 4								# Interval to upload predicted evaluation run
	eval_save_all_plots: bool = False								# Save all plots for evaluation

	image_x_y_start_end: Tuple[int, int, int, int] = (60, 110, 420, 390)	# x_start, y_start, x_end, y_end of the image crop
	image_crop_size: Tuple[int, int] = (144, 144)			# Crop size of the input image
	random_shift_augmentation_padding: int = 6							# Use random shift augmentation method

	plot_infos: List[PlotInfo]= field(default_factory=lambda: [
		{"start_episode": 0, "start_step": 30, "end_episode": 0, "end_step": 90},
		{"start_episode": 1, "start_step": 30, "end_episode": 1, "end_step": 90},
		{"start_episode": 2, "start_step": 30, "end_episode": 2, "end_step": 90},
		{"start_episode": 3, "start_step": 30, "end_episode": 3, "end_step": 90},
		
		{"start_episode": 0, "start_step": 0, "end_episode": 0, "end_step": 200},
		{"start_episode": 1, "start_step": 0, "end_episode": 1, "end_step": 200},
		{"start_episode": 2, "start_step": 0, "end_episode": 2, "end_step": 200},
		{"start_episode": 3, "start_step": 0, "end_episode": 3, "end_step": 200},

		{"start_episode": 0, "start_step": 50, "end_episode": 100, "end_step": 50},

		{"start_episode": 0, "start_step": 0, "end_episode": 100, "end_step": 200},
	])

	act: Config = None
	backbone: str = "resnet18"
	start_time: str = datetime.now().strftime("%d-%m %H-%M")
	device: str = ""
	save_path: str = ""
	wandb_run_name: str = ""
	
	@classmethod
	def from_json(cls, file_path: str) -> "RLConfig":
		with open(os.path.join(file_path, "rl_config.json"), "r") as f:
			data = json.load(f)
		return cls(**data)
			
	def __repr__(self):
		fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
		return f"{self.__class__.__name__}({fields})"