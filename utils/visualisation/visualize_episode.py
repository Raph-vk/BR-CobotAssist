import os
import numpy as np
import cv2
import h5py
import glob
import random
from datetime import datetime
from tqdm import tqdm

import matplotlib.pyplot as plt

import IPython
e = IPython.embed

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]



def get_most_recent_dataset_dir(parent_dir):
    """Find the most recent subdirectory in the parent directory."""
    subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    if not subdirs:
        raise ValueError(f"No subdirectories found in {parent_dir}")
    most_recent_dir = max(subdirs, key=os.path.getmtime)
    return most_recent_dir

def get_random_hdf5(dataset_dir):
    """Find random hdf5 file in the given directory."""
    hdf5_files = glob.glob(os.path.join(dataset_dir, "*.hdf5"))
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {dataset_dir}")
    random_hdf5 = random.choice(hdf5_files)
    return os.path.splitext(os.path.basename(random_hdf5))[0]


def main(dataset_name=None):
    # Set dataset_dir to the correct path (sibling to tos_app)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tos_app_data'))
    try:
        if dataset_name:
            # If a dataset name is provided, use the most recent subdir and look for that file
            dataset_dir = get_most_recent_dataset_dir(parent_dir)
            selected_dataset = dataset_name
        else:
            # Find most recent subdir, then most recent hdf5 file in it
            dataset_dir = get_most_recent_dataset_dir(parent_dir)
            selected_dataset = get_random_hdf5(dataset_dir)
            print(f"Using most recent dataset: {selected_dataset} in {dataset_dir}")
    except Exception as e:
        print(f"Error finding dataset: {str(e)}")
        return

    # Define output paths
    video_path = os.path.join(dataset_dir, selected_dataset + '_video.mp4')
    plot_path = os.path.join(dataset_dir, selected_dataset + '_joint_pos.png')
    # Ensure output directory exists
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Load data from HDF5
    action_master, action_puppet, joint_states, action, color_image_dict, depth_image_dict = load_hdf5(dataset_dir, selected_dataset)
    # Visualize joints
    visualize_joints(action_master, action_puppet, joint_states, action, plot_path=plot_path)
    # Save video with both color and depth images
    save_videos(color_image_dict, depth_image_dict, video_path=video_path)
    


def save_videos(color_videos, depth_videos, video_path=None):
    """
    Save videos from color and depth cameras side by side.

    Parameters:
    - color_videos (dict): Dictionary of color camera_name: images array (n_frames, h, w, 3)
    - depth_videos (dict): Dictionary of depth camera_name: images array (n_frames, h, w)
    - dt (float): Time delta between frames
    - video_path (str): Path to save the video
    """
    if not isinstance(color_videos, dict) or not isinstance(depth_videos, dict):
        print("Color and depth videos should be dictionaries.")
        return

    # Ensure both dictionaries have the same camera names
    color_cam_names = sorted(color_videos.keys())
    depth_cam_names = sorted(depth_videos.keys())
    if color_cam_names != depth_cam_names:
        print("Mismatch in camera names between color and depth videos.")
        return

    cam_names = color_cam_names
    num_cams = len(cam_names)

    # Check if there are any cameras
    if num_cams == 0:
        print("No cameras found in the dataset.")
        return

    # Ensure all cameras have the same number of frames
    num_frames = color_videos[cam_names[0]].shape[0]
    for cam_name in cam_names:
        if color_videos[cam_name].shape[0] != depth_videos[cam_name].shape[0]:
            print(f"Frame count mismatch for camera {cam_name}.")
            return

    # Assuming all cameras have the same resolution
    first_color = color_videos[cam_names[0]][0]
    first_depth = depth_videos[cam_names[0]][0]

    h_color, w_color, _ = first_color.shape
    h_depth, w_depth = first_depth.shape[:2]  # Support single and multi-channel

    # Initialize video writer
    fps = int(62.5)
    # Total width: (color + depth) for each camera
    total_width = num_cams * (w_color + w_depth)
    # Total height: max of color and depth heights
    total_height = max(h_color, h_depth)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (total_width, total_height))

    for frame_idx in tqdm(range(num_frames), desc="Saving video"):
        frame_images = []
        for cam_name in cam_names:
            color_img = color_videos[cam_name][frame_idx]
            depth_img = depth_videos[cam_name][frame_idx]

            
            # swap B and R channel
            color_img = color_img[:, :, [2, 1, 0]]

            # Diagnostic prints for the first frame
            if frame_idx == 0:
                print(f"Processing camera: {cam_name}")
                print(f"Original color image shape: {color_img.shape}, dtype: {color_img.dtype}")
                print(f"Original depth image shape: {depth_img.shape}, dtype: {depth_img.dtype}")

            # Handle multi-channel depth images by converting to single-channel if necessary
            if len(depth_img.shape) == 3:
                print(f"Depth image {cam_name} has {depth_img.shape[2]} channels. Converting to single-channel.")
                depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
                print(f"Converted depth image shape: {depth_img.shape}")

            # Normalize depth image to 0-255
            depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
            
            # Convert to uint8
            depth_norm_uint8 = depth_norm.astype(np.uint8)
            
            # Apply color map
            try:
                depth_colored = cv2.applyColorMap(depth_norm_uint8, cv2.COLORMAP_JET)
            except cv2.error as e:
                print(f"Error applying color map to depth image {cam_name} at frame {frame_idx}: {e}")
                # Create a black image as a fallback
                depth_colored = np.zeros((h_depth, w_depth, 3), dtype=np.uint8)
            
            # Resize depth image if necessary to match color image
            if depth_colored.shape[:2] != color_img.shape[:2]:
                depth_colored = cv2.resize(depth_colored, (color_img.shape[1], color_img.shape[0]))
                print(f"Resized depth image {cam_name} to match color image size.")

            # Concatenate color and depth images vertically
            combined = np.concatenate((color_img, depth_colored), axis=1)
            frame_images.append(combined)

        # Concatenate all camera pairs horizontally
        final_frame = np.concatenate(frame_images, axis=1)
        out.write(final_frame)

    out.release()
    print(f'Saved video to: {video_path}')



def visualize_joints(master_positions, send_position_robots, robot_positions, action, plot_path=None, ylim=None, label_overwrite=None):
    master_positions = np.array(master_positions)
    send_position_robots = np.array(send_position_robots)
    robot_positions = np.array(robot_positions)
    action = np.array(action) if action is not None else None

    num_ts, num_dim = robot_positions.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(robot_positions[:, dim_idx], label='robot_positions')
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    if send_position_robots is not None:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.plot(send_position_robots[:, dim_idx], label='send_position_robots')
            ax.legend()

    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(master_positions[:, dim_idx], label='master_positions')
        ax.legend()

    if action is not None:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.plot(action[:, dim_idx], label='action')
            ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    if plot_path:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        print(f'Saved joint_pos plot to: {plot_path}')
    plt.close()

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    print(f'Loading dataset from: {dataset_path}')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        print(f"Top-level keys in HDF5: {list(root.keys())}")
        # Try new format first
        if 'master_positions' in root:
            print("Using new HDF5 format.")
            action_master = root['master_positions'][()]
            print(f"master_positions shape: {action_master.shape}, dtype: {action_master.dtype}")
            if 'send_position_robots' in root:
                action_puppet = root['send_position_robots'][()]
                print(f"send_position_robots shape: {action_puppet.shape}, dtype: {action_puppet.dtype}")
            else:
                action_puppet = None
                print("send_position_robots not found in root.")
            joint_states = root['robot_positions'][()]
            print(f"robot_positions shape: {joint_states.shape}, dtype: {joint_states.dtype}")
            action = None  # Not present in new format
            # Load color images
            color_image_dict = {}
            depth_image_dict = {}
            if 'images' in root:
                print(f"Image camera keys: {list(root['images'].keys())}")
                for cam_name in root['images']:
                    cam_group = root['images'][cam_name]
                    print(f"  Camera: {cam_name}, keys: {list(cam_group.keys())}")
                    if 'color' in cam_group:
                        color_image_dict[cam_name] = cam_group['color'][()]
                        print(f"    color shape: {color_image_dict[cam_name].shape}, dtype: {color_image_dict[cam_name].dtype}")
                    if 'depth' in cam_group:
                        depth_image_dict[cam_name] = cam_group['depth'][()]
                        print(f"    depth shape: {depth_image_dict[cam_name].shape}, dtype: {depth_image_dict[cam_name].dtype}")
            else:
                print("No 'images' group found in root.")
    return action_master, action_puppet, joint_states, action, color_image_dict, depth_image_dict




if __name__ == '__main__':
    # Set dataset_name to None to use the most recent dataset, or set to a string to use a specific one
    dataset_name = ""  # e.g., 'episode_001' or None
    main(dataset_name)
