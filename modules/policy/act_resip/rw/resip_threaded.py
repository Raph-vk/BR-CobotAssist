import os
import time
import torch
import wandb
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import kornia.augmentation as aug

from .running_norm import RunningNorm
from diffusers.optimization import get_scheduler

from .ResiP_utils.utils import calculate_advantage
from .ResiP_utils.rl_models import ResidualMLPPolicy
from .backbone_loader.main import build_ACT_model_and_optimizer, get_args_parser

class ResiP():
    """
    Residual‑policy that adds residual action to the frozen ACT baseline action and trains
    itself on‑policy with PPO whenever enough new rollouts are buffered.
    """

    def __init__(self, 
                 cfg,
                 shared_memories,
                 shm_lock,
                 stop_resip_event,
                 interrupt_detectron_event,
                 ckpt_path=None):
        self.cfg = cfg
        self.logger = self.setup_logger(getattr(cfg, "verbose_logging", True))
        self.global_step = 0
        self.current_episode = 0
        self.current_step = 0
        self.current_episode_reward_set = False
        self.prediction_times = []
        self.last_action_gripper_state = 0
        self.current_action_gripper_state = 0
        
        (   self.qpos_shm,
            self.vision_data_shm,
            self.next_base_action_shm,
            self.final_action_shm,
            self.predict_action_flag_shm,
            self.action_ready_flag_shm,
            self.activate_detectron_flag_shm,
            self.reward_flag_shm,
            self.env_reset_flag_shm
        ) = shared_memories
        self.shm_lock = shm_lock
        self.stop_resip_event = stop_resip_event
        self.interrupt_detectron_event = interrupt_detectron_event

        self.wandb_run = wandb.init(project=f"TOS_{self.cfg.algorithm}_{self.cfg.act.environment}", mode=self.cfg.wandb_mode)
        self.cfg.wandb_run_name = "test" if self.cfg.wandb_mode == "offline" else self.wandb_run.name
        if (self.cfg.save_path == ""):
            save_path = os.path.join(os.getcwd(), "real-world", "resip-models", self.cfg.start_time + "_" + self.cfg.wandb_run_name)
            self.cfg.save_path = save_path
            os.makedirs(self.cfg.save_path, exist_ok=True)  # Create the directory if it doesn't exist  

        # Setup logger
        self.verbose_logging = self.cfg.verbose_logging
        self.logger = logging.getLogger("ResiP Process")
        if self.verbose_logging:
            logging.basicConfig(level=logging.INFO, format='[%(process)d %(name)s] %(message)s')
        else:
            logging.basicConfig(level=logging.CRITICAL)  # Only log critical errors

        self.obs_shape = self.cfg.qpos_shape[0] + self.cfg.act.vision_space * self.cfg.num_cameras * 2 + self.cfg.action_shape[0]

        if self.cfg.random_shift_augmentation_padding:
            self.random_shift = nn.Sequential(
                nn.ReplicationPad2d(self.cfg.random_shift_augmentation_padding),
                aug.RandomCrop(self.cfg.vision_shape[1:])
            )

        self.backbones = self.build_backbones()

        self.model = ResidualMLPPolicy(
            self.obs_shape,
            self.cfg.action_shape[0],
            self.cfg
        ).to(self.cfg.device)

        self.opt_actor = optim.AdamW(
            self.model.actor_parameters, 
            lr=self.cfg.actor_lr, 
            betas=self.cfg.betas, 
            eps=self.cfg.eps, 
            weight_decay=self.cfg.weight_decay
        )
        self.lr_scheduler_actor = get_scheduler(
            name="cosine", 
            optimizer=self.opt_actor, 
            num_warmup_steps=self.cfg.actor_num_warmup_steps, 
            num_training_steps=self.cfg.optimizer_steps
        )
        self.opt_critic = optim.AdamW(
            self.model.critic_parameters, 
            lr=self.cfg.critic_lr, 
            eps=self.cfg.eps, 
            weight_decay=self.cfg.weight_decay
        )
        self.lr_scheduler_critic = get_scheduler(
            name="cosine", 
            optimizer=self.opt_critic, 
            num_warmup_steps=self.cfg.critic_num_warmup_steps, 
            num_training_steps=self.cfg.optimizer_steps
        )

        self._reset_episode_buffers()
        self.qpos_norm = RunningNorm(*self.cfg.qpos_shape, self.cfg)
        self.vision_norm = [RunningNorm((self.cfg.vision_shape[0],), self.cfg) for _ in range(self.cfg.num_cameras)]            
  
        if (ckpt_path is not None):
            self._load_checkpoint(ckpt_path)
            self.logger.info(f"Loaded checkpoint from {ckpt_path}")
            print("qpos state dict", self.qpos_norm.state_dict())

    def run(self):
        self.logger.info(f"[ResiP] Started")
        while not self.stop_resip_event.is_set():
            self.logger.info(f"[ResiP] stop resip event is not set, continuing...")
            while True:
                with self.shm_lock:
                    env_reset_flag = np.ndarray(self.cfg.flag_shape, dtype=np.uint8, buffer=self.env_reset_flag_shm.buf)[0]
                    predict_action_flag = np.ndarray(self.cfg.flag_shape, dtype=np.uint8, buffer=self.predict_action_flag_shm.buf)[0]
                self.logger.info(f"[ResiP] env_reset_flag: {env_reset_flag}, predict_action_flag: {predict_action_flag}")
                if (env_reset_flag):
                    time.sleep(0.01)
                    continue
                elif predict_action_flag:
                    np.ndarray(self.cfg.flag_shape, dtype=np.uint8, buffer=self.predict_action_flag_shm.buf)[0] = 0
                    break    
                time.sleep(0.001)

            self.logger.info(f"Starting work on Episode:Step {self.current_episode}:{self.current_step}")
            self._predict_next_action()
            self._observe()
            self.logger.info(f"Observed environment")
            if self.current_step == self.cfg.episode_steps:
                with self.shm_lock:          
                    np.ndarray(self.cfg.flag_shape, dtype=np.uint8, buffer=self.env_reset_flag_shm.buf)[0] = 1

                if (((self.current_episode + 1) % self.cfg.update_episodes) == 0 and self.current_episode != 0):
                    self._train_ppo()
                    self._reset_episode_buffers()
                    self.current_episode = 0  
                    self.logger.info(f"Ending train Loop")
                    self._save_checkpoint()
                else:    
                    self.current_episode += 1

                self.current_step = 0
                self.current_episode_reward_set = False
                self.last_action_gripper_state = 0
                self.current_action_gripper_state = 0
                if self.prediction_times:
                    self.logger.info(f"Average prediction time: {round(sum(self.prediction_times) / len(self.prediction_times), 2)} ms")
                    self.logger.info(f"Average prediction time last 10 steps: {round(sum(self.prediction_times[-10:]) / len(self.prediction_times[-10:]), 2)} ms")
                    self.logger.info(f"Max prediction time last 10 steps: {round(max(self.prediction_times[-10:]), 2)} ms")

        self.wandb_run.finish()
        self.logger.info(f"Shutting down")

    def _reset_episode_buffers(self):
        self.observations = torch.zeros((self.cfg.episode_steps, self.cfg.update_episodes, self.obs_shape), dtype=torch.float32, device=self.cfg.device)
        self.next_base_actions = torch.zeros((self.cfg.episode_steps, self.cfg.update_episodes, *self.cfg.action_shape), dtype=torch.float32, device=self.cfg.device)
        self.residual_actions = torch.zeros((self.cfg.episode_steps, self.cfg.update_episodes, *self.cfg.action_shape), dtype=torch.float32, device=self.cfg.device)
        self.logprobs = torch.zeros((self.cfg.episode_steps, self.cfg.update_episodes), dtype=torch.float32, device=self.cfg.device)
        self.state_values = torch.zeros((self.cfg.episode_steps, self.cfg.update_episodes), dtype=torch.float32, device=self.cfg.device)
        self.advantages = torch.zeros((self.cfg.episode_steps, self.cfg.update_episodes), dtype=torch.float32, device=self.cfg.device)
        self.returns = torch.zeros((self.cfg.episode_steps, self.cfg.update_episodes), dtype=torch.float32, device=self.cfg.device)
        self.rewards = torch.zeros((self.cfg.episode_steps, self.cfg.update_episodes), dtype=torch.float32, device=self.cfg.device)
        self.dones = torch.zeros((self.cfg.episode_steps, self.cfg.update_episodes), dtype=torch.float32, device=self.cfg.device)
        self.next_state_values = torch.zeros((self.cfg.update_episodes), dtype=torch.float32, device=self.cfg.device)  
        self.next_done = torch.zeros((self.cfg.update_episodes), dtype=torch.uint8, device=self.cfg.device)

    def _predict_next_action(self):
        start_time = time.time()
        with self.shm_lock:
            qpos = torch.from_numpy(np.copy(np.ndarray(self.cfg.qpos_shape, dtype=np.float32, buffer=self.qpos_shm.buf))).to(self.cfg.device)
            vision_data = torch.from_numpy(np.copy(np.ndarray((self.cfg.num_cameras, *self.cfg.vision_shape), dtype=np.float32, buffer=self.vision_data_shm.buf))).to(self.cfg.device)
            next_base_action = torch.from_numpy(np.copy(np.ndarray(self.cfg.action_shape, dtype=np.float32, buffer=self.next_base_action_shm.buf))).to(self.cfg.device)
        
        features = self._vision_feats(vision_data)
        if self.cfg.normalize:
            qpos, vision_data = self._normalize_observations(qpos, vision_data)
        residual_obs = torch.cat([qpos.unsqueeze(0), features.reshape(-1).unsqueeze(0), next_base_action.unsqueeze(0)], dim=-1)

        if (self.current_step != 0 and (self.current_step % self.cfg.episode_steps) == 0):
            next_state_value = self.model.get_state_value(residual_obs).reshape(1, -1)
            self.next_state_values[self.current_episode] = next_state_value
        else:
            with torch.no_grad():
                residual_action, logprob, entropy, state_value, action_mean = \
                    self.model.get_action_and_value(residual_obs)
            action = (next_base_action + residual_action * self.cfg.alpha).squeeze(0)
            action[-1] = 1 if action[-1].item() >= self.cfg.gripper_active_threshold else 0
            self.last_action_gripper_state = self.current_action_gripper_state
            self.current_action_gripper_state = action[-1].item()

            with self.shm_lock:
                np.ndarray(self.cfg.action_shape, dtype=np.float32, buffer=self.final_action_shm.buf)[:] = action.cpu().numpy()
                np.ndarray(self.cfg.flag_shape, dtype=np.uint8, buffer=self.action_ready_flag_shm.buf)[0] = 1

            self.observations[self.current_step, self.current_episode] = residual_obs
            self.next_base_actions[self.current_step, self.current_episode] = next_base_action
            self.residual_actions[self.current_step, self.current_episode] = residual_action
            self.logprobs[self.current_step, self.current_episode] = logprob
            self.state_values[self.current_step, self.current_episode] = state_value

            self.prediction_times.append(round((time.time()-start_time) * 1000, 2))
    
    def _observe(self):
        with self.shm_lock:
            reward_flag = np.ndarray(self.cfg.flag_shape, dtype=np.uint8, buffer=self.reward_flag_shm.buf)[0]
            qpos = torch.from_numpy(np.copy(np.ndarray(self.cfg.qpos_shape, dtype=np.float32, buffer=self.qpos_shm.buf))).to(self.cfg.device)
            np.ndarray(self.cfg.flag_shape, dtype=np.uint8, buffer=self.reward_flag_shm.buf)[0] = 0

        # self.logger.info(f"Received reward flag: {reward_flag}")
        # self.logger.info(f"last_action_gripper_state: {self.last_action_gripper_state}, current_action_gripper_state: {self.current_action_gripper_state}")
        if (self.last_action_gripper_state == 1 and self.current_action_gripper_state == 0):
            with self.shm_lock:
                np.ndarray(self.cfg.flag_shape, dtype=np.uint8, buffer=self.activate_detectron_flag_shm.buf)[0] = 1
            # self.logger.info(f"set activate_detectron_flag_shm to 1")
        
        if (qpos[-2].item() < 3 and qpos[-2].item() > 357):
            if (not self.current_episode_reward_set):
                self.rewards[self.current_step, self.current_episode] = reward_flag

        if (self.rewards[self.current_step, self.current_episode] > 0):
            self.logger.info(f"Received reward: {self.rewards[self.current_step, self.current_episode]}")
        self.dones[self.current_step, self.current_episode] = self.next_done[self.current_episode]
        self.next_done[self.current_episode] = self.next_done[self.current_episode] | reward_flag
        
        if ((self.current_step + 1) % self.cfg.episode_steps == 0 and self.current_step != 0):
            if self.cfg.truncation_as_done:
                self.next_done[self.current_episode] = 1
        
        if reward_flag > 0:
            self.current_episode_reward_set = True
        
        self.current_step += 1
        self.global_step += 1

    def _train_ppo(self):
        self.rewards *= self.cfg.reward_scaling_factor
        advantages, returns = calculate_advantage(
            self.state_values,
            self.next_state_values,
            self.rewards,
            self.dones,
            self.next_done,
            self.cfg.episode_steps,
            self.cfg.discount_factor,
            self.cfg.gae_lambda)       
        
        batch_size = self.cfg.episode_steps * self.cfg.update_episodes
        b_obs = self.observations.reshape([batch_size, -1])
        b_actions = self.residual_actions.reshape([batch_size, -1])
        b_logprobs = self.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.state_values.reshape(-1)

        loss_names = ["total_loss", "pg_loss_1", "pg_loss_2", "pg_loss", "v_loss", "entropy_loss", "ppo_loss", "residual_l1_loss", "residual_l2_loss", "policy_loss"]
        loss_buffers = {name: [] for name in loss_names}

        for epoch in range(self.cfg.update_epochs):
            b_inds = torch.randperm(batch_size, device=self.cfg.device)
            for start in range(0, batch_size, batch_size):
                end = start + batch_size

                mb_inds = b_inds[start:end]
                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_logprobs = b_logprobs[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_values = b_values[mb_inds]

                loss, approx_kl_value = self.model.compute_loss(
                    mb_obs,
                    mb_actions,
                    mb_logprobs,
                    mb_advantages,
                    mb_returns,
                    mb_values,
                    loss_buffers,
                    self.cfg
                )

                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt_actor.step()
                self.opt_critic.step()

                if (self.cfg.kl_threshold is not None and approx_kl_value > self.cfg.kl_threshold):
                    return

    def build_backbones(self):
        args = get_args_parser().parse_args([])  # or provide the correct args
        tmp_model, tmp_optimizer = build_ACT_model_and_optimizer(args, self.cfg)
        # print(tmp_model.state_dict().keys())
        backbone_path = "/home/tos-pc3/Desktop/tos_app/modules/policy/act_resip/rw/backbone_loader/model.ckpt"
        # 2. Load the checkpoint
        checkpoint = torch.load(backbone_path, map_location="cpu")
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']

        # Remove 'model.' prefix from keys
        stripped_state_dict = {
            k.replace('model.', ''): v for k, v in checkpoint.items()
        }

        tmp_model.load_state_dict(stripped_state_dict, strict=False)


        # 4. Access the backbones
        backbones = tmp_model.backbones
        # del tmp_model  # Clean up the temporary model

        for backbone in backbones:
            backbone.to(self.cfg.device)
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.eval()   

        return backbones
    
    def _normalize_observations(self, qpos, vision_data, eval=False):
        if not eval:
            self.qpos_norm.update(qpos)
        qpos = self.qpos_norm.normalize(qpos)

        normed_vision = list(map(
            lambda args: (
                args[1].update(args[0], dims=(1, 2)) if not eval else None,
                args[1].normalize(args[0], dims=(1, 2)).to(self.cfg.device)
            )[1],
            zip(vision_data, self.vision_norm)
        ))
        vision_data = torch.stack(normed_vision, dim=0).to(self.cfg.device)
        return qpos, vision_data
    
    def _vision_feats(self, vision_data):
        streams = [torch.cuda.Stream(device=self.cfg.device) for _ in range(self.cfg.num_cameras)]
        results = [None] * self.cfg.num_cameras

        for idx in range(self.cfg.num_cameras):
            if self.random_shift:
                vision_data[idx] = self.random_shift(vision_data[idx].unsqueeze(0)).squeeze(0)

        for idx in range(self.cfg.num_cameras):
            with torch.cuda.stream(streams[idx]):
                with torch.no_grad():
                    feature, _ = self.backbones[idx](vision_data[idx])
                results[idx] = feature[0][0].mean(dim=[1, 2])

        torch.cuda.synchronize(self.cfg.device)
        features = torch.stack(results)
        return features

    def _save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'opt_actor_state_dict': self.opt_actor.state_dict(),
            'opt_critic_state_dict': self.opt_critic.state_dict(),
            'lr_scheduler_actor_state_dict': self.lr_scheduler_actor.state_dict(),
            'lr_scheduler_critic_state_dict': self.lr_scheduler_critic.state_dict(),
            'global_step': self.global_step,
            'cfg': self.cfg,
            'qpos_norm_state': self.qpos_norm.state_dict(),
            'vision_norm_state': [vn.state_dict() for vn in self.vision_norm]
        }
        torch.save(checkpoint, self.cfg.save_path + f"/episode_{int(self.global_step / self.cfg.episode_steps)}.pth")
        self.logger.info(f"Checkpoint saved to {self.cfg.save_path}")

    def _load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, weights_only=False, map_location=self.cfg.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt_actor.load_state_dict(checkpoint['opt_actor_state_dict'])
        self.opt_critic.load_state_dict(checkpoint['opt_critic_state_dict'])
        self.lr_scheduler_actor.load_state_dict(checkpoint['lr_scheduler_actor_state_dict'])
        self.lr_scheduler_critic.load_state_dict(checkpoint['lr_scheduler_critic_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.qpos_norm.load_state_dict(checkpoint['qpos_norm_state'])
        for vn, state in zip(self.vision_norm, checkpoint['vision_norm_state']):
            vn.load_state_dict(state)

        self.cfg.save_dir = os.path.join(self.cfg.save_path, str(datetime.now().strftime("%d-%m %H-%M")) + "_ckpt")
        self.logger.info(f"Checkpoint loaded from {ckpt_path}")

    def setup_logger(self, verbose_logging):
        logger = logging.getLogger("ResiP Process")
        if verbose_logging:
            logging.basicConfig(level=logging.INFO, format='[%(process)d %(name)s] %(message)s')
        else:
            logging.basicConfig(level=logging.CRITICAL)
        return logger