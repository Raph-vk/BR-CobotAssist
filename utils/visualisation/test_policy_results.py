import os
import random
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt

import os
import random
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt

def test_policy_chunk_prediction(
    dataset_dir,
    policy,
    stats,
    camera_names,
    device,
    episodes_to_plot,
    chunk_size
):
    """
    For each of several episodes, pick ONE random timestamp t. 
    Use the policy to predict a chunk of future actions (length chunk_size).
    Compare and plot those predictions against the ground-truth actions.
    Display color images at timestamp t side by side.
    Includes debug prints to confirm color/depth normalization is consistent
    with training.
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

    # Put model in eval mode on the correct device
    policy.eval().to(device)

    # Gather all HDF5 files
    all_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')])
    if not all_files:
        print(f"No .hdf5 files found in {dataset_dir}")
        return

    # Example: restrict to a single episode or do your custom logic
    # all_files = ['episode_501.hdf5']  # Remove/modify if you want more episodes
    episodes_to_plot = min(episodes_to_plot, len(all_files))
    selected_episodes = random.sample(all_files, episodes_to_plot)

    for ep_index, filename in enumerate(selected_episodes):
        fpath = os.path.join(dataset_dir, filename)
        with h5py.File(fpath, 'r') as h5f:
            # Debug: Print HDF5 structure
            print(f"\nDebugging HDF5 structure for {filename}:")
            print(f"Root keys: {list(h5f.keys())}")
            
            # Updated paths to match new save interface structure
            # Check if we have the new structure first, fallback to old if needed
            if 'master_positions' in h5f:
                print("Using new HDF5 structure")
                # New structure
                action_master = h5f['master_positions'][()]  # Master positions as actions
                # For joint_states, we need to combine robot_position and master_position
                robot_positions = h5f['robot_positions'][()]
                master_positions = h5f['master_positions'][()]
                # Combine them to create joint_states (assuming 21:28 was master, rest robot)
                # Create joint_states by concatenating robot_positions and master_positions
                joint_states = np.concatenate([robot_positions, master_positions], axis=1)
                print(f"Data shapes - robot_positions: {robot_positions.shape}, master_positions: {master_positions.shape}")
            else:
                print("Using old HDF5 structure")
                # Fallback to old structure if it exists
                action_master = h5f['/observations/action_master'][()]
                joint_states = h5f['/observations/joint_states'][()]
                print(f"Data shapes - action_master: {action_master.shape}, joint_states: {joint_states.shape}")
            
            episode_len = len(action_master)

            # Prepare color & depth references - updated paths
            color_images_dict = {}
            depth_images_dict = {}
            
            for cam in camera_names:
                if 'images' in h5f and cam in h5f['images']:
                    # New structure: /images/{cam}/color and /images/{cam}/depth
                    print(f"Camera {cam}: using new structure /images/{cam}/")
                    color_images_dict[cam] = h5f[f'images/{cam}/color']
                    depth_images_dict[cam] = h5f[f'images/{cam}/depth']
                elif '/observations/images' in h5f and cam in h5f['/observations/images']:
                    # Fallback to old structure
                    print(f"Camera {cam}: using old structure /observations/images/{cam}")
                    color_images_dict[cam] = h5f[f'/observations/images/{cam}']
                    depth_images_dict[cam] = h5f[f'/observations/depth_images/{cam}']
                else:
                    print(f"Warning: Camera {cam} not found in either structure!")
                    # Check what's available
                    if 'images' in h5f:
                        print(f"Available cameras in /images/: {list(h5f['images'].keys())}")
                    if '/observations' in h5f and 'images' in h5f['/observations']:
                        print(f"Available cameras in /observations/images/: {list(h5f['/observations/images'].keys())}")
                    continue

            print(f"\n[Episode {ep_index}] {filename} has length {episode_len}")
            valid_max_t = episode_len - chunk_size
            if valid_max_t <= 0:
                print("Episode too short for chunked predictions. Skipping.")
                continue

            # Pick random or fixed timestamp t
            t = random.randint(0, valid_max_t - 1)
            # t = 300  # Example override if you want a fixed time
            print(f"Selected timestamp t={t} for chunk prediction.")

            # 1) Extract the current joint state
            # For new structure, use robot_positions directly as current joint state
            # For old structure, extract from joint_states[21:28]
            if 'master_positions' in h5f:
                # New structure: robot_positions contains the actual joint positions
                raw_joints = robot_positions[t].copy()
            else:
                # Old structure: extract from joint_states
                raw_joints = joint_states[t, 21:28].copy()
                
            print('  Current joint state:', raw_joints)
            print('  Corresponding action_master:', action_master[t])
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
                print(f"\nCamera={cam}, shape c_img={c_img.shape}, d_img={d_img.shape}")
                print(f"  c_img range => min={c_img.min()}, max={c_img.max()}")
                print(f"  d_img range => min={d_img.min()}, max={d_img.max()}")

                # **Normalize** color and depth
                c_img /= 255.0
                d_img /= 65535.0
                # make depth images 0 
                # d_img = np.zeros_like(d_img)

                # Debug prints: min/max after normalization
                print(f"  c_img normalized => min={c_img.min():.4f}, max={c_img.max():.4f}")
                print(f"  d_img normalized => min={d_img.min():.4f}, max={d_img.max():.4f}")

                # Concatenate => (H, W, 4)
                combined = np.concatenate([c_img, d_img], axis=-1)
                color_imgs.append(combined)  # Or rename color_imgs-> combined_imgs if you'd prefer

            # 3) Stack across cameras => (num_cams, H, W, 4)
            all_cam_images = np.stack(color_imgs, axis=0)

            # Debug shape
            print(f"Stacked camera images shape => {all_cam_images.shape} (num_cams, H, W, 4)")

            # 4) Move to Torch: reorder => (num_cams, 4, H, W)
            #    then unsqueeze => (1, num_cams, 4, H, W)
            image_data = (
                torch.from_numpy(all_cam_images)
                .permute(0, 3, 1, 2)
                .contiguous()
                .unsqueeze(0)
                .to(device)
            )
            print(f"Final image_data shape => {image_data.shape} (1, num_cams, 4, H, W)\n")

            # 5) Run the policy to predict a chunk
            with torch.no_grad():
                outputs = policy(input_joints, image_data, None, None)

            # 6) Extract predicted actions from policy outputs
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
                print(f"Unexpected output shape from policy: {out_actions.shape}")
                print("Skipping plotting for this episode.")
                continue

            # 7) Compare predicted actions vs ground-truth
            actual_actions = action_master[t : t + chunk_size]
            print(f"\nGround-truth actions shape => {actual_actions.shape}")
            print(f"Predicted actions shape => {predicted_actions.shape}\n")

            # Plot results or store them. Below is the same logic as your original
            num_cams = len(camera_names)
            fig_height = 3 + 2 * 7
            fig, axs = plt.subplots(nrows=8, ncols=num_cams, figsize=(4*num_cams, 20))
            plt.subplots_adjust(wspace=0.1, hspace=0.4)
            fig.suptitle(
                f"Episode {ep_index}: Chunk Prediction @ t={t}\n({filename})", fontsize=16
            )

            # 7a) Show the color images (the original color frames, ignoring depth channel for display)
            for cam_idx, cam in enumerate(camera_names):
                ax = axs[0, cam_idx] if num_cams > 1 else axs[0]
                # Show just the color part as uint8 (if you want to visualize the original image)
                # or show the normalized float if you prefer. We'll cast it back to uint8 for display:
                show_img = (color_images_dict[cam][t].astype(np.uint8))
                ax.imshow(show_img)
                ax.axis('off')
                ax.set_title(f"{cam} at t={t}")

            # 7b) Plot each joint's predicted vs actual
            # If only 1 camera, axs is 1D
            if num_cams == 1:
                axs = axs.reshape(-1, 1)

            # Ensure both arrays have the same length for plotting
            min_len = min(chunk_size, predicted_actions.shape[0], actual_actions.shape[0])
            x = np.arange(min_len)
            for j_idx in range(7):
                for cam_idx, cam in enumerate(camera_names):
                    ax = axs[j_idx + 1, cam_idx] if num_cams > 1 else axs[j_idx + 1]
                    ax.plot(x, predicted_actions[:min_len, j_idx], 'r.-', label=f'Pred j{j_idx}')
                    ax.plot(x, actual_actions[:min_len, j_idx], 'b.--', label=f'Act j{j_idx}')
                    ax.set_ylabel(f'Joint {j_idx}')
                    if j_idx == 0:
                        ax.set_title("Pred vs Actual")
                    ax.legend(loc='best')

            for ax in axs[-1]:
                ax.set_xlabel("Time-step in chunk")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            plt.close(fig)

            print(f"Chunk prediction plotted for Episode {ep_index} at t={t}.")


if __name__ == "__main__":
    import sys
    import os
    import torch

    # Add the project root to the Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    
    import pickle
    from modules.policy.act.detr.models import ACTPolicy  # Updated import path

    # ----------------------------------------------------------------------
    # Example usage
    # ----------------------------------------------------------------------
    ckpt_dir = os.path.expanduser("~/tos_app_data/20250630_155645/Models/20250630_160425")
    ckpt_path = os.path.join(ckpt_dir, "policy_best_epoch_66.ckpt")
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")

    # Load stats
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # Define your policy configuration
    policy_config = {
        "backbone": "resnet34",
        "camera_names": [
            "cam_1",
            "cam_2"
        ],
        "chunk_size": 75,
        "ckpt_dir": "/home/teun/tos_app_data/20250630_155645/Models/20250630_160425",
        "dec_layers": 8,
        "dim_feedforward": 1024,
        "enc_layers": 6,
        "hidden_dim": 1024,
        "kl_weight": 100,
        "lr": 1e-05,
        "lr_backbone": 1e-05,
        "nheads": 32,
        "num_queries": 75
    }

    # Initialize the policy
    policy = ACTPolicy(policy_config)
    policy.load_state_dict(torch.load(ckpt_path))
    policy.cuda()

    # Directory with episodes
    dataset_dir = os.path.expanduser("~/tests/second_test")

    # Cameras used in the dataset - updated to match config
    camera_names = ["cam_1", "cam_2"]

    # Run chunk prediction test
    test_policy_chunk_prediction(
        dataset_dir=dataset_dir,
        policy=policy,
        stats=stats,
        camera_names=camera_names,
        device='cuda',
        episodes_to_plot=5,
        chunk_size=50  # Define the chunk size
    )
