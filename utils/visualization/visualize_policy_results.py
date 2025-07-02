import os
import random
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import json
import cv2
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.visualization.visualize_robot import RobotVisualizer 

def test_policy_chunk_prediction(
    dataset_dir,
    policies,
    stats,
    camera_names,
    device,
    episodes_to_plot,
    chunk_size,
    episode_number=None,
    timestamp=None,
    plot_teachbot=True,
    plot_robotdemo=True
):
    """
    For each of several episodes, pick ONE random timestamp t. 
    Use multiple policies to predict chunks of future actions (length chunk_size).
    Compare and plot those predictions against the ground-truth actions.
    Display color images at timestamp t side by side.
    
    Args:
        policies: dict of {model_name: policy} pairs
    """

    # -----------------------
    # Helper Functions
    # -----------------------
    def pre_process_joints(joints):
        """Normalize joint positions."""
        return (joints - stats['joint_pos_mean']) / stats['joint_pos_std']

    def post_process_action(a):
        """Denormalize action outputs."""
        return a * stats['action_std'] + stats['action_mean']

    # Put all models in eval mode on the correct device
    for model_name, policy in policies.items():
        policy.eval().to(device)

    # Gather all HDF5 files
    all_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')],key=lambda fn: int(fn.split('_')[1].split('.')[0]))
    if not all_files:
        print(f"No .hdf5 files found in {dataset_dir}")
        return
    
    # Select episodes to plot
    if episode_number is None:
        episodes_to_plot = min(episodes_to_plot, len(all_files))
        selected_episodes = random.sample(all_files, episodes_to_plot)
    else:
        selected_episodes = [f'episode_{episode_number}.hdf5']

    for ep_index, filename in enumerate(selected_episodes):
        fpath = os.path.join(dataset_dir, filename)
        with h5py.File(fpath, 'r') as h5f:
            action_master = h5f['/observations/action_master'][()]
            joint_states = h5f['/observations/joint_states'][()]
            episode_len = len(action_master)

            # Prepare color & depth references
            color_images_dict = {cam: h5f[f'/observations/images/{cam}'] for cam in camera_names}
            depth_images_dict = {cam: h5f[f'/observations/depth_images/{cam}'] for cam in camera_names}

            print(f"\n[Episode {ep_index}] {filename} has length {episode_len}")
            valid_max_t = episode_len - chunk_size
            if valid_max_t <= 0:
                print("Episode too short for chunked predictions. Skipping.")
                continue

            # Pick random or fixed timestamp t
            if timestamp is None:
                t = random.randint(0, valid_max_t - 1)
            else:
                t = timestamp
                if t < 0 or t >= valid_max_t:
                    t = random.randint(0, valid_max_t - 1)
                    print(f"Invalid timestamp {timestamp} for episode length {episode_len} and chunk size {chunk_size}. Sampling random t={t}.")
        
            print(f"Selected timestamp t={t} for chunk prediction.")

            # 1) Extract the current joint state
            raw_joints = joint_states[t, 21:28].copy()
            # print('  Current joint state:', raw_joints, type(raw_joints))
            # print('  Corresponding action_master:', action_master[t], type(action_master[t]))
            input_joints = pre_process_joints(raw_joints)
            input_joints = torch.tensor(input_joints, dtype=torch.float32, device=device).unsqueeze(0)

            # 2) Prepare images for time t
            color_imgs, depth_imgs = [], []
            for cam in camera_names:
                # shape (H, W, 3)
                c_img = color_images_dict[cam][t]  
                # shape (H, W) or (H, W, 1)
                d_img = depth_images_dict[cam][t]

                # Force shape to (H, W, 1) for depth
                if len(d_img.shape) == 2:
                    d_img = d_img[..., np.newaxis]

                # Convert to float32
                c_img = c_img.astype(np.float32)
                d_img = d_img.astype(np.float32)

                # Debug prints: min/max before normalization
                # print(f"\nCamera={cam}, shape c_img={c_img.shape}, d_img={d_img.shape}")
                # print(f"  c_img range => min={c_img.min()}, max={c_img.max()}")
                # print(f"  d_img range => min={d_img.min()}, max={d_img.max()}")

                # **Normalize** color and depth
                c_img /= 255.0
                d_img /= 65535.0
                # make depth images 0 
                # d_img = np.zeros_like(d_img)

                # Debug prints: min/max after normalization
                # print(f"  c_img normalized => min={c_img.min():.4f}, max={c_img.max():.4f}")
                # print(f"  d_img normalized => min={d_img.min():.4f}, max={d_img.max():.4f}")

                # Concatenate => (H, W, 4)
                combined = np.concatenate((c_img, d_img), axis=-1)
                color_imgs.append(combined)  # Or rename color_imgs-> combined_imgs if you'd prefer

            # 3) Stack across cameras => (num_cams, H, W, 4)
            all_cam_images = np.stack(color_imgs, axis=0)

            # Debug shape
            # print(f"Stacked camera images shape => {all_cam_images.shape} (num_cams, H, W, 4)")

            # 4) Move to Torch: reorder => (num_cams, 4, H, W)
            #    then unsqueeze => (1, num_cams, 4, H, W)
            image_data = (
                torch.from_numpy(all_cam_images)
                .permute(0, 3, 1, 2)
                .contiguous()
                .unsqueeze(0)
                .to(device)
            )
            # print(f"Final image_data shape => {image_data.shape} (1, num_cams, 4, H, W)\n")

            # 5) Run all policies to predict chunks
            all_predictions = {}
            for model_name, policy in policies.items():
                with torch.no_grad():
                    outputs = policy(input_joints, image_data, None, None)
                
                # Extract predicted actions from policy outputs
                if isinstance(outputs, (tuple, list)):
                    out_actions = outputs[0].cpu().numpy()
                else:
                    out_actions = outputs.cpu().numpy()

                # out_actions should be shape: (1, chunk_size, 7) or (chunk_size, 7)
                # Post-process accordingly
                if out_actions.ndim == 3 and out_actions.shape[0] == 1 and out_actions.shape[2] == 7:
                    predicted_actions = post_process_action(out_actions[0])  # shape (chunk_size, 7)
                elif out_actions.ndim == 2 and out_actions.shape[0] == chunk_size and out_actions.shape[1] == 7:
                    predicted_actions = post_process_action(out_actions)
                else:
                    print(f"Unexpected output shape from policy {model_name}: {out_actions.shape}")
                    continue
                
                all_predictions[model_name] = predicted_actions

            # Skip if no valid predictions
            if not all_predictions:
                print("No valid predictions from any policy. Skipping plotting for this episode.")
                continue

            # 6) Visualize the results
            first_prediction = next(iter(all_predictions.values()))
            nj       = first_prediction.shape[1]       # number of joints (7)
            nc       = len(camera_names)           # number of cameras (4)

            # --- build one Figure + GridSpec ----------------------
            rows = nj + 2                # 7 joint rows + 1 RGB row + 1 Depth row
            cols = nc + 2                # one column per camera for the images + 2 for robot joint pos visualization

            # give the two image‐rows a bit more vertical real‐estate
            height_ratios = [1.8,1.8]+ [1]*nj 
            width_ratios  = [1]*cols      # all image columns same width

            fig = plt.figure(figsize=(4*nc, nj*1.2 + 2), num=f"{filename[:-5]} Chunk t={t}")
            
            gs  = fig.add_gridspec(rows, cols,
                                height_ratios=height_ratios,
                                width_ratios=width_ratios,
                                hspace=0.3, wspace=0.1,
                                bottom=0.1, top=0.9)

            fig.suptitle(f"Action Chunk Prediction of {filename} @ t = {t} = {t/62.5}s", fontsize=18)

            # --- 1) joint‐plots, each spanning all 4 columns ----------
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']  # Colors for different policies
            skip = 0
            for j in range(nj):
                if j == 3:
                    print("Skipping joint 3 in plots, since this joint is disabled.")
                    skip = 1
                    continue  # skip the gripper joint (index 3) if not needed
                ax = fig.add_subplot(gs[j-skip+2, :])   # all columns
                
                # plot predictions from all policies
                for idx, (model_name, predicted_actions) in enumerate(all_predictions.items()):
                    color = colors[idx % len(colors)]
                    ax.plot(np.arange(chunk_size), predicted_actions[:, j], 
                           color=color, linestyle='-', marker='.', 
                           label=f'{model_name}')
                
                # plot the demonstration actions of the teachbot or robot if wanted
                if plot_teachbot:
                    teachbot_actions = action_master[t : t + chunk_size]
                    ax.plot(np.arange(chunk_size), teachbot_actions[:, j],   'k.--', label=f'Teachbot')
                if plot_robotdemo:
                    robotdemo_actions = joint_states[t : t + chunk_size, 21:28].copy()
                    ax.plot(np.arange(chunk_size), robotdemo_actions[:, j],   'gray', linestyle='--', marker='.', label=f'Robot demo')
                
                ax.set_ylabel(f'Joint {j}',  loc='center', labelpad=20)
                if j == 0:
                    ax.legend(loc='upper right')
                if j == nj-1:
                    ax.set_xlabel("Time-step in chunk")

            # --- 2) RGB images on row = 0, columns 0..nc-1 -----------
            for i, cam in enumerate(camera_names):
                ax = fig.add_subplot(gs[0, i])
                rgb = color_images_dict[cam][t].astype(np.uint8)
                ax.imshow(rgb)
                ax.set_title(f"{cam}")
                ax.axis('off')

            # --- 3) Depth images on row = 1, columns 0..nc-1 ------
            for i, cam in enumerate(camera_names):
                ax = fig.add_subplot(gs[1, i])
                depth = depth_images_dict[cam][t]
                # ensure single‐channel
                if depth.ndim == 3:
                    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
                # normalize → [0–255] + color‐map
                d8     = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                d_col  = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
                # resize if needed
                h, w = rgb.shape[:2]
                if d_col.shape[:2] != (h, w):
                    d_col = cv2.resize(d_col, (w, h))
                ax.imshow(d_col)
                ax.axis('off')

            # --- 4) Robot joint positions on last column --------------
            # Initialize Meshcat robot visualizer
            # print(f'input joints {raw_joints}')
            visualizer = RobotVisualizer()
            robot_front, robot_side = visualizer.move_joints(np.array(raw_joints).reshape(1,7), extract_frame=True)
    
            ax = fig.add_subplot(gs[0:2, 4])
            height = robot_front.shape[0]
            min_height = round(height/2) - 350
            max_height = round(height/2) + 150
            width = robot_front.shape[1]
            min_width = round(width/2) - 250
            max_width = round(width/2) + 250
            robot_front_cropped = robot_front[min_height:max_height, min_width:max_width, :3]
            ax.imshow(robot_front_cropped)  # Show RGB part of the robot visualizer, and crop image
            ax.set_title(f"Robot Front View")
            ax.axis('off')

            ax = fig.add_subplot(gs[0:2, 5])
            robot_side_cropped = robot_side[min_height:max_height, min_width:max_width, :3]
            ax.imshow(robot_side_cropped)  # Show RGB part of the robot visualizer, and crop image
            ax.set_title(f"Robot Side View")
            ax.axis('off')

            plt.show()

            print(f"Chunk prediction plotted for {filename} at t={t}.")


if __name__ == "__main__":
    import pickle
    from modules.policy.act.detr.models.act_policy import ACTPolicy 

    test = "second_test"
    models = ["20250522_all_cameras"] 
    # models = ["20250522_all_cameras", "20252705_masked"]  # List of models to compare

    dataset_dir = os.path.expanduser(f"~/tos_app_data/{test}")
    
    # Load multiple policies
    policies = {}
    all_stats = {}
    
    for model in models:
        print(f"Loading model: {model}")
        ckpt_dir = os.path.join(dataset_dir, f"Models/{model}")
        
        # Find the best checkpoint or epoch checkpoint
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith('policy_') and f.endswith('.ckpt')]
        if not ckpt_files:
            print(f"No checkpoint files found for model {model}")
            continue
            
        # Prefer best epoch checkpoint
        best_ckpt = [f for f in ckpt_files if 'best_epoch' in f]
        if best_ckpt:
            ckpt_path = os.path.join(ckpt_dir, best_ckpt[0])
        else:
            # Use the latest epoch checkpoint
            epoch_ckpts = [f for f in ckpt_files if 'epoch_' in f and 'best' not in f]
            if epoch_ckpts:
                # Sort by epoch number
                epoch_ckpts.sort(key=lambda x: int(x.split('_')[2]))
                ckpt_path = os.path.join(ckpt_dir, epoch_ckpts[-1])
            else:
                print(f"No suitable checkpoint found for model {model}")
                continue
        
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        json_path = os.path.join(ckpt_dir, "config.json")
        
        # Check if required files exist
        if not all(os.path.exists(p) for p in [ckpt_path, stats_path, json_path]):
            print(f"Missing required files for model {model}")
            continue

        # Load stats
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        all_stats[model] = stats

        # Load JSON config
        with open(json_path, 'r') as f:
            config = json.load(f)
        policy_config = config["policy_config"]

        # Initialize the policy
        policy = ACTPolicy(policy_config)
        policy.load_state_dict(torch.load(ckpt_path))
        policy.cuda()
        
        policies[model] = policy
        print(f"Successfully loaded model: {model}")
    
    if not policies:
        print("No policies were successfully loaded!")
        exit(1)
    
    # Use stats from the first model (assuming they're similar)
    first_model_name = next(iter(all_stats.keys()))
    stats = all_stats[first_model_name]
    
    # Use config from the first model for camera names and chunk size
    first_model = next(iter(policies.keys()))
    ckpt_dir = os.path.join(dataset_dir, f"Models/{first_model}")
    json_path = os.path.join(ckpt_dir, "config.json")
    with open(json_path, 'r') as f:
        config = json.load(f)
    policy_config = config["policy_config"]

    print(f"Comparing {len(policies)} models: {list(policies.keys())}")

    # Run chunk prediction test
    test_policy_chunk_prediction(
        dataset_dir=dataset_dir,
        policies=policies,
        stats=stats,
        camera_names=policy_config["camera_names"],
        device='cuda',
        episodes_to_plot=1,
        chunk_size= policy_config["chunk_size"],  # Define the chunk size
        # episode_number=101,
        # timestamp=215
    )
