import os
import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, config, logger):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.config = config
        self.logger = logger

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

            # Get joint data - these are the action targets (send_position_robot)
            actions = root['send_position_robots'][()].astype(np.float32)  # Action targets (includes gripper as last joint)
            robot_positions = root['robot_positions'][()].astype(np.float32)  # Robot observations (includes gripper as last joint)  
            timestamps = root['robot_position_timestamps'][()].astype(np.float32)
            
            # Images are sampled every record_divisor steps
            # So image[n] corresponds to joint_state[n * record_divisor]
            num_images = len(root[f'images/{self.camera_names[0]}/color'])
            episode_len = len(actions)
        
            # Choose a random start point, but ensure we have aligned image data
            max_image_start = max(0, num_images - 1) 
            image_start_idx = np.random.choice(max_image_start + 1)
            
            # Calculate corresponding joint state index
            joint_start_idx = image_start_idx * record_divisor
            
            # Make sure we don't go beyond available joint states
            if joint_start_idx >= episode_len:
                joint_start_idx = episode_len - 1
                image_start_idx = joint_start_idx // record_divisor

            # Retrieve joint states (robot position at the selected timestep)
            # joint_pos now includes gripper as last element: [j1, j2, j3, j4, j5, j6, gripper]
            joint_pos = robot_positions[joint_start_idx]
            
            # Make sure joint 4 is always 0 when predicting (j4 is at index 3)
            j4_locked = self.config["general"]["j4_locked"]
            if j4_locked and len(joint_pos) > 3:
                joint_pos[3] = 0

            # Retrieve action data from the selected start point
            # action now includes gripper as last element: [j1, j2, j3, j4, j5, j6, gripper]
            action = actions[joint_start_idx:]
            action_len = episode_len - joint_start_idx

            # ------------------------------------------------
            # 1) Load color images for each camera - using new structure
            # ------------------------------------------------
            color_images = []
            for cam_name in self.camera_names:
                # New structure: images/{cam_name}/color
                c_img = root[f'images/{cam_name}/color'][image_start_idx]
                color_images.append(c_img)
            # shape: (num_cams, H, W, 3)
            color_images = np.stack(color_images, axis=0)

            # ------------------------------------------------
            # 2) Load depth images for each camera - using new structure  
            # ------------------------------------------------
            depth_images = []
            for cam_name in self.camera_names:
                # New structure: images/{cam_name}/depth
                d_img = root[f'images/{cam_name}/depth'][image_start_idx]
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

        # Debug final shape
        # print("Final image_data shape:", tuple(image_data.shape), "\n")

        # ------------------------------------------------
        # The rest is unchanged from your original logic
        # ------------------------------------------------
        # Create padded action array with the original episode length
        padded_action = np.zeros((episode_len, action.shape[1]), dtype=np.float32)
        padded_action[joint_start_idx:joint_start_idx + action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[joint_start_idx + action_len:] = 1

        joint_pos_data = torch.from_numpy(joint_pos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # Normalize joint_pos and action
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        joint_pos_data = (joint_pos_data - self.norm_stats["joint_pos_mean"]) / self.norm_stats["joint_pos_std"]

        return image_data, joint_pos_data, action_data, is_pad



def get_norm_stats(dataset_dir, num_episodes):
    all_joint_pos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        print('given episodes:', num_episodes, '    | loaded:', episode_idx+1)
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        try:
            with h5py.File(dataset_path, 'r') as root:
                # Use new structure: robot_positions and send_position_robots
                robot_positions = root['robot_positions'][()].astype(np.float32)  # Joint positions (observations)
                actions = root['send_position_robots'][()].astype(np.float32)     # Actions (targets)
                
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
        "example_joint_pos": robot_positions[0] if len(robot_positions) > 0 else None
    }

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, config, logger):
    print(f'\nData from: {dataset_dir}\n')

    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    train_ratio = 0.7
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, config, logger)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, config, logger)
    
    num_workers = 4
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
