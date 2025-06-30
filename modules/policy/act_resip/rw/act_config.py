from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from typing import List

@dataclass
class ACTConfig:
	# Define your config attributes with default values
	algorithm: str = "ACT"
	environment: str = "relocate"
	sim_environment: str = "AdroitHandRelocate-v1"
	action_space: int = 7
	joint_space: int = 7
	env_space: int = 9
	vision_space: int = 256
	render_mode: str = "rgb_array"									# "human", "rgb_array" or None
	vision_based: bool = True										# Use visual inputs for training
	environment_state_based: bool = False							# Use environment state information for training
	same_train_val_set: bool = False								# Enable to set evalution set equal to train set
	backbone: str = "resnet34"										# Set backbone, either resnet18 or resnet34
	max_episode_steps: int = 200									# Max number of episode steps to predict	
	eval_trials: int = 100											# Trials used for evaluation
	eval_interval: int = 5										# Epochs between evaluating
	eval_visualize_interval: int = 12								# Interval to upload predicted evaluation run
	save_interval: int = 5										# Interval between saved models
	env_success_reward: int = 3000									# Reward required for evaluation to be considered a success
	reward_save_threshold: int = 30									# Threshold from which best reward will be saved
	eval_rollout_chunk: bool = True									# Use action chunking rollouts for evaluation
	random_shift_augmentation: bool = True							# Use random shift augmentation method
	dataset_name: str = "relocate/expert-init-vision-v30"			# Dataset name
	chunk_size: int = 5												# Size of predicted action chunks
	train_ratio: float = 0.9										# Train- and Evaluation split
	batch_size_train: int = 1000
	batch_size_val: int = 1
	epochs: int = 300
	alpha: float = 0.5												# Trade-off factor between L1- and MSE-loss
	kl_weight: int = 10												# Kullback–Leibler weighting factor
	policy_lr: float = 1e-4											# Policy starting lr
	policy_backbone_lr: float = 1e-5								# Policy backbone lr
	policy_lr_eta_min: float = 1e-6									# Policy ending lr using cosine annealing lr scheduler
	policy_wd: float = 1e-5											# Policy weigt decay
	latent_dim: int = 32											# Latent variable dimension
	d_model: int = 256												# Hidden dimension ACT
	dropout: float = 0.1											# Dropout rate
	num_head: int = 8												# Number of transformer heads
	dim_feedforward: int = 1024										# Feedforward dimension
	num_encoder_layers: int = 4										# Number of encoder layers
	num_decoder_layers: int = 4										# Number of decoder layers
	activation: str = "relu"										# Activation Function
	device: str = "cpu"
	start_time: str = datetime.now().strftime("%d-%m %H-%M")
	save_path: str = ""
	wandb_run_name: str = ""
	train_indices: List[int] = field(default_factory=list)
	val_indices: List[int] = field(default_factory=list)
	
	### Below Unused
	camera_names: List[str] = field(default_factory=list)
	ckpt_dir: str = ""
	num_queries: int = 50
	task_name: str = ""
	### Above Unused

	@classmethod
	def from_json(cls, file_path: str) -> "Config":
		with open(os.path.join(file_path, "config.json"), "r") as f:
			data = (json.load(f))["policy_config"]
		return cls(**data)
			
	def __repr__(self):
		fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
		return f"{self.__class__.__name__}({fields})"

