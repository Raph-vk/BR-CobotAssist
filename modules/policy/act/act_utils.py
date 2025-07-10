import os
import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, config, logger, recording_speed=1.0, robot_speed=1.0):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.config = config
        self.logger = logger
        self.recording_speed = float(recording_speed)
        self.robot_speed = float(robot_speed)
        self.speed_factor = self.robot_speed / self.recording_speed

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            # Get metadata to understand the structure
            try:
                record_divisor = root['metadata'].attrs['record_divisor']
            except (KeyError, AttributeError):
                record_divisor = 4  # Default fallback
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Metadata 'record_divisor' not found in episode {episode_id}. Using default value {record_divisor}.")

            # Get joint data - these are the action targets (sent_robot_position)
            teachbot_positions = root['teachbot_positions'][()].astype(np.float32)  # Action targets (includes gripper as last joint)
            robot_positions = root['robot_positions'][()].astype(np.float32)  # Robot observations (includes gripper as last joint)  
            timestamps = root['robot_position_timestamps'][()].astype(np.float32)
            
            # Apply record_divisor first (base sampling rate)
            # This aligns joint data with image sampling rate
            teachbot_positions = teachbot_positions[::record_divisor]
            robot_positions = robot_positions[::record_divisor]
            
            # Apply speed factor subsampling if needed
            if self.speed_factor > 1:
                # Choose random start offset for subsampling to get different sub-sequences
                speed_step = int(self.speed_factor)
                start_offset = np.random.choice(speed_step)
                teachbot_positions = teachbot_positions[start_offset::speed_step]
                robot_positions = robot_positions[start_offset::speed_step]
                # Store the offset for later image indexing
                self._current_speed_offset = start_offset
                self._current_speed_step = speed_step
            else:
                self._current_speed_offset = 0
                self._current_speed_step = 1
            
            # Now teachbot_positions and robot_positions are aligned with the effective sampling rate
            original_actions_shape = teachbot_positions.shape
            episode_len = len(teachbot_positions)
            
            # Choose a random start point within the subsampled data
            max_start = max(0, episode_len - 1) if episode_len > 0 else 0
            start_idx = np.random.choice(max_start + 1) if max_start > 0 else 0

            # Retrieve joint states (robot position at the selected timestep)
            joint_pos = robot_positions[start_idx]
            
            # Make sure joint 4 is always 0 when predicting (j4 is at index 3)
            j4_locked = self.config["general"]["j4_locked"]
            if j4_locked and len(joint_pos) > 3:
                joint_pos[3] = 0

            # Retrieve action data from the selected start point
            actions = teachbot_positions[start_idx:]
            action_len = episode_len - start_idx

            # ------------------------------------------------
            # 1) Load color images for each camera - using new structure
            # ------------------------------------------------
            color_images = []
            for cam_name in self.camera_names:
                # Calculate the actual image index based on subsampling
                # start_idx is within the subsampled sequence
                # We need to map it back to the original image sequence
                actual_image_idx = self._current_speed_offset + (start_idx * self._current_speed_step)
                c_img = root[f'images/{cam_name}/color'][actual_image_idx]
                color_images.append(c_img)
            # shape: (num_cams, H, W, 3)
            color_images = np.stack(color_images, axis=0)

            # ------------------------------------------------
            # 2) Load depth images for each camera - using new structure  
            # ------------------------------------------------
            depth_images = []
            for cam_name in self.camera_names:
                # Use the same actual image index
                actual_image_idx = self._current_speed_offset + (start_idx * self._current_speed_step)
                d_img = root[f'images/{cam_name}/depth'][actual_image_idx]
                if len(d_img.shape) == 2:
                    d_img = d_img[..., np.newaxis]  # shape: (H, W, 1)
                depth_images.append(d_img)
            # shape: (num_cams, H, W, 1)
            depth_images = np.stack(depth_images, axis=0)


            # ------------------------------------------------
            # 3) Combine color+depth channel-wise => (H, W, 4) per camera
            # ------------------------------------------------
            combined_images = []
            for i in range(len(self.camera_names)):
                # shape: (H, W, 3) + (H, W, 1) => (H, W, 4)
                combined = np.concatenate([color_images[i], depth_images[i]], axis=-1)
                combined_images.append(combined)
            # shape: (num_cams, H, W, 4)
            all_cam_images = np.stack(combined_images, axis=0)

        # ------------------------------------------------
        # Debug prints - shapes after merging color+depth
        # ------------------------------------------------
        all_cam_images = all_cam_images.astype(np.float32)

        # Split the channels for debugging
        color_part = all_cam_images[..., :3]  # shape: (num_cams, H, W, 3)
        depth_part = all_cam_images[..., 3:]  # shape: (num_cams, H, W, 1)

        # Normalize color ([0,255] => [0,1])
        color_part /= 255.0

        # Normalize depth ([0,65535] => [0,1])
        depth_part /= 65535.0

        # Re-combine them back into (num_cams, H, W, 4)
        all_cam_images = np.concatenate([color_part, depth_part], axis=-1)

        # ------------------------------------------------
        # 5) Move to torch and reorder => (num_cams, 4, H, W)
        # ------------------------------------------------
        image_data = torch.from_numpy(all_cam_images)
        image_data = image_data.permute(0, 3, 1, 2)  # shape: (num_cams, 4, H, W)

        # ------------------------------------------------
        # The rest is unchanged from your original logic
        # ------------------------------------------------

        # Create padded action array with the original episode length
        padded_action = np.zeros(original_actions_shape, dtype=np.float32)
        padded_action[:action_len] = actions
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        joint_pos_data = torch.from_numpy(joint_pos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # Normalize joint_pos and action
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        joint_pos_data = (joint_pos_data - self.norm_stats["joint_pos_mean"]) / self.norm_stats["joint_pos_std"]

        return image_data, joint_pos_data, action_data, is_pad



def get_norm_stats(dataset_dir, num_episodes, recording_speed, robot_speed):
    # Ensure speeds are floats
    recording_speed = float(recording_speed)
    robot_speed = float(robot_speed)
    
    all_joint_pos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        print('given episodes:', num_episodes, '    | loaded:', episode_idx+1)
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        try:
            with h5py.File(dataset_path, 'r') as root:
                robot_positions = root['robot_positions'][()].astype(np.float32)  # Joint positions (observations)
                actions = root['teachbot_positions'][()].astype(np.float32)     # Actions (targets)
                record_divisor = root['metadata'].attrs.get('record_divisor', 4)  # Default to 4 if not found
                
                factor = robot_speed / recording_speed
                if factor < 1:
                    print(f"Robot speed {robot_speed} is lower than recording speed {recording_speed}. Not possible")
                    continue
                elif factor > 1:
                    # Need to subsample: robot executes faster than recorded
                    # factor=2 means take every 2nd image/joint pair  
                    # This creates multiple sub-sequences from the same episode
                    print(f"Robot speed {robot_speed} is higher than recording speed {recording_speed}. Subsampling by factor {factor}")
                    
                    # Start with the base sampling (every record_divisor)
                    base_robot_positions = robot_positions[::record_divisor]
                    base_actions = actions[::record_divisor]
                    
                    # Now subsample by the speed factor
                    speed_step = int(factor)
                    for start_offset in range(speed_step):
                        sub_robot_positions = base_robot_positions[start_offset::speed_step]
                        sub_actions = base_actions[start_offset::speed_step]
                        
                        if len(sub_robot_positions) > 0:  # Only add if we have data
                            all_joint_pos_data.append(torch.from_numpy(sub_robot_positions))
                            all_action_data.append(torch.from_numpy(sub_actions))
                    
                    # Skip the normal append at the end since we already added the subsampled data
                    continue
                    
                elif int(factor) == 1:
                    print(f"Robot speed {robot_speed} is equal to recording speed {recording_speed}. No downsampling needed")
                    robot_positions = robot_positions[::record_divisor]
                    actions = actions[::record_divisor]

            all_joint_pos_data.append(torch.from_numpy(robot_positions))
            all_action_data.append(torch.from_numpy(actions))
        except Exception as e:
            print('ERROR episode', episode_idx, e)
            break

    all_joint_pos_data = torch.stack(all_joint_pos_data)
    all_action_data = torch.stack(all_action_data)

    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    joint_pos_mean = all_joint_pos_data.mean(dim=[0, 1], keepdim=True)
    joint_pos_std = all_joint_pos_data.std(dim=[0, 1], keepdim=True)
    joint_pos_std = torch.clip(joint_pos_std, 1e-2, np.inf)

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "joint_pos_mean": joint_pos_mean.numpy().squeeze(),
        "joint_pos_std": joint_pos_std.numpy().squeeze(),
        "example_joint_pos": robot_positions
    }

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, config, logger, recording_speed, robot_speed):
    print(f'\nData from: {dataset_dir}\n')

    # Ensure speeds are floats
    recording_speed = float(recording_speed)
    robot_speed = float(robot_speed)

    norm_stats = get_norm_stats(dataset_dir, num_episodes, recording_speed, robot_speed)
    train_ratio = 0.7
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, config, logger, recording_speed, robot_speed)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, config, logger, recording_speed, robot_speed)
    
    num_workers = 1
    prefetch_factor = 5
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=num_workers, prefetch_factor=prefetch_factor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=num_workers, prefetch_factor=prefetch_factor)

    return train_dataloader, val_dataloader, norm_stats


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
