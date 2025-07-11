import numpy as np
import os
import yaml
import cv2
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

from visualize_robot import RobotVisualizer

def main(dataset_dir, episode_idx, yaml_path = 'utils/visualization/config_visualization.yaml',
         save_plot=False,
         save_plot_dropoff=False, 
         save_plot_all_dropoff=False, 
         save_video=False,
         visualize_robot=False,
         save_robot_video=False):
    dataset_name = f'episode_{episode_idx}'

    # Load inputs from yaml file
    with open(yaml_path, 'r') as file:
        settings = yaml.safe_load(file)
    state_names = settings['state_names']
    DT = settings['dt_plot']
    print(f"Loaded plot settings from: {os.path.abspath(yaml_path)}")

    # Load data from HDF5
    action_master, action_puppet, joint_states, action, color_image_dict, depth_image_dict = load_hdf5(dataset_dir, dataset_name)
    
    # Plot joint graphs
    if save_plot:
        plot_path = create_path(dataset_dir,'graphs', dataset_name, '_joint_pos.png')
        save_joints_graph(plot_path, action_master, action_puppet, joint_states, action, state_names)

    # Plot joint graphs sorted by dropoff
    if save_plot_dropoff:
        plot_dropoff_path = create_path(dataset_dir,'graphs', dataset_name, '_joint_pos_dropoff.png')
        save_joints_dropoff(plot_dropoff_path, joint_states, state_names)

    # Plot all episodes joint graphs sorted by dropoff
    if save_plot_all_dropoff:
        plot_all_dropoff_path = create_path(dataset_dir,'graphs', dataset_name, '_joint_pos_all_dropoff.png')
        save_joints_dropoff(plot_all_dropoff_path, joint_states, state_names, all_episodes=True, all_episodes_dir=dataset_dir)

    # Save video with both color and depth images
    if save_video:
        video_path = create_path(dataset_dir, 'videos', dataset_name, '_video.mp4')
        save_videos(video_path, color_image_dict, depth_image_dict, DT)

    # Initialize Meshcat robot visualizer
    if visualize_robot or save_robot_video:
        video_robot_path = create_path(dataset_dir, 'videos', dataset_name, '_robot_video.mp4')
        visualizer = RobotVisualizer()
        visualizer.move_joints(joint_states[:, 21:28], videopath=video_robot_path, recording=save_robot_video)

def create_path(dataset_dir, folder_name, dataset_name, ending):
    path = os.path.join(dataset_dir, folder_name, dataset_name + ending)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        print(f'Created directory: {os.path.dirname(path)}')
    return path

def load_hdf5(dataset_dir, dataset_name, only_joint_states=False):
    if dataset_name.endswith('.hdf5'):
        dataset_path = os.path.join(dataset_dir, dataset_name)
        print(f'Loading joint states from: {dataset_path}')
    else:
        dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
        print(f'Loading dataset from: {dataset_path}')

    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    if only_joint_states:
        with h5py.File(dataset_path, 'r') as root:
            joint_states = root['/observations/joint_states'][()]
            return joint_states

    else:
        with h5py.File(dataset_path, 'r') as root:
            action_master = root['/observations/action_master'][()]
            action_puppet = root['/observations/action_puppet'][()]
            joint_states = root['/observations/joint_states'][()]
            action = root['/action'][()]
            
            # Load color images
            color_image_dict = {}
            cam_names = list(root[f'/observations/images/'].keys())
            # print(f'Camera names: {cam_names}')
            # cam_names = cam_names[2:] # Skip the first two cameras
            # print(f'Camera names: {cam_names}')
            # for cam_name in root[f'/observations/images/'].keys():
            for cam_name in cam_names:
                color_image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
            
            # Load depth images
            depth_image_dict = {}
            # for depth_cam_name in root[f'/observations/depth_images/'].keys():
            for depth_cam_name in cam_names:
                depth_image_dict[depth_cam_name] = root[f'/observations/depth_images/{depth_cam_name}'][()]
        
        return action_master, action_puppet, joint_states, action, color_image_dict, depth_image_dict

def save_joints_graph(plot_path, action_master, action_puppet, joint_states, action, joint_names, ylim=None, label_overwrite=None):
    joint_pos = joint_states[:, 21:28]

    joint_pos = np.array(joint_pos) # ts, dim
    action_puppet = np.array(action_puppet)
    action_master = np.array(action_master)
    action = np.array(action)
    
    num_ts, num_dim = joint_pos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(joint_pos[:, dim_idx], label='robot')
        ax.set_title(f'Joint {dim_idx}: {joint_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(action_puppet[:, dim_idx], label='buffer robot')
        ax.legend()

    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(action_master[:, dim_idx], label='teachbot')
        ax.legend()

    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(action[:, dim_idx], label='action')
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved joint angle plot to: {plot_path}')
    plt.close()
  
def save_joints_dropoff(plot_path, joint_states, joint_names, dropoff=True, all_episodes=False, all_episodes_dir=None):

    def init_plot(joint_state):
        joint_pos = joint_state[:, 21:28]
        joint_pos = np.array(joint_pos)
        num_ts, num_dim = joint_pos.shape
        h, w = 2, num_dim
        num_figs = num_dim
        fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))
        return axs
    
    def get_all_episodes_joint_states(dataset_dir):
        print(f'Loading all episodes joint states from: {dataset_dir}')
        joint_states = []
        files_unsorted = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
        files = sorted(files_unsorted,key=lambda fn: int(fn.split('_')[1].split('.')[0]))
        for f in files:
            joint_states.append(load_hdf5(dataset_dir, f, only_joint_states=True))
        return joint_states

    def visualize_single_episode(joint_state, joint_names, axs, dropoff):
        joint_pos = joint_state[:, 21:28]
        joint_pos = np.array(joint_pos) # ts, dim
        num_ts, num_dim = joint_pos.shape
 
        # Look for dropoff indices
        dropoffs = [i for i in range(1, len(joint_pos[:, 6])) if joint_pos[i-1, 6] == 1 and joint_pos[i, 6] == 0] # For dropoff so from vacuum on to off
        vacuumons = [i for i in range(1, len(joint_pos[:, 6])) if joint_pos[i-1, 6] == 0 and joint_pos[i, 6] == 1] # For vacuumon so from vacuum off to on

        allpoints = dropoffs + vacuumons
        allpoints.sort()

        # delete vacuumon points that are too close to the next dropoff point
        for i in range(len(allpoints)-1):
            if allpoints[i+1] - allpoints[i] < 65:
                # print("!!!!!!!!!!!!!!!!!!  ",allpoints[i], allpoints[i+1])
                try:
                    vacuumons.remove(allpoints[i])
                    # print(f"excluded {allpoints[i]} from vacuumon")
                except:
                    pass

        # plot joint state
        if dropoff:
            for dropoff in dropoffs:
                for dim_idx in range(num_dim):
                    ax = axs[dim_idx]
                    ax.plot(joint_pos[dropoff:, dim_idx], label='joint_pos')
                    ax.set_title(f'Joint {dim_idx}: {joint_names[dim_idx]}')
                    # ax.legend()
        else:
            for dropoff in vacuumons:
                for dim_idx in range(num_dim):
                    ax = axs[dim_idx]
                    ax.plot(joint_pos[dropoff:, dim_idx], label='joint_pos')
                    ax.set_title(f'Joint {dim_idx}: {joint_names[dim_idx]}')
                    # ax.legend()

    axs = init_plot(joint_states)
    if all_episodes:
        # Gather all joint states from all episodes
        joint_states = get_all_episodes_joint_states(all_episodes_dir)
        for joint_state in joint_states:
            visualize_single_episode(joint_state, joint_names, axs, dropoff=dropoff)
    else:
        visualize_single_episode(joint_states, joint_names, axs, dropoff=dropoff)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved joint angle by dropoff plot to: {plot_path}')
    plt.close()

def save_videos(video_path, color_videos, depth_videos = {}, dt=62.5, include_depth=True, roi_points=None, save_one_frame=False):
    """
    Save videos from color and depth cameras side by side.

    Parameters:
    - color_videos (dict): Dictionary of color camera_name: images array (n_frames, h, w, 3)
    - depth_videos (dict): Dictionary of depth camera_name: images array (n_frames, h, w)
    - dt (float): Time delta between frames
    - video_path (str): Path to save the video
    - roi_mask (numpy array): Optional mask for region of interest
    - roi_points (list): Optional points defining the region of interest
    """

    if not isinstance(color_videos, dict):
        print("Color videos should be dictionaries.")
        return
    
    if include_depth and not isinstance(depth_videos, dict):
        print("Depth videos should be dictionaries.")
        return

    # Ensure both dictionaries have the same camera names
    color_cam_names = sorted(color_videos.keys())
    if include_depth:
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
    if include_depth:
        for cam_name in cam_names:
            if color_videos[cam_name].shape[0] != depth_videos[cam_name].shape[0]:
                print(f"Frame count mismatch for camera {cam_name}.")
                return

    # Assuming all cameras have the same resolution
    first_color = color_videos[cam_names[0]][0]
    h_color, w_color, _ = first_color.shape

    if include_depth:
        first_depth = depth_videos[cam_names[0]][0]
        h_depth, w_depth = first_depth.shape[:2]  # Support single and multi-channel


    # Initialize video writer
    fps = int(1/dt)
    # Total width: (color + depth) for each camera, Total height: max of color and depth heights
    if include_depth:
        total_width = num_cams * (w_color + w_depth)
        total_height = max(h_color, h_depth)
    else:
        total_width = num_cams * w_color
        total_height = h_color

    # Make mask for region of interest
    if roi_points is not None:
        mask = np.zeros(color_videos[cam_names[0]][0][:, :, [2, 1, 0]].shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(roi_points)], 255)
    
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (total_width, total_height))

    for frame_idx in tqdm(range(num_frames), desc="Saving video"):
        frame_images = []
        for cam_name in cam_names:
            color_img = color_videos[cam_name][frame_idx]
            # swap B and R channel
            color_img = color_img[:, :, [2, 1, 0]]

            if save_one_frame and frame_idx == 0:
                fig_path = os.path.join(os.path.dirname(video_path), f'{cam_name}_frame_{frame_idx}.png')
                cv2.imwrite(fig_path, color_img)
                print(f"Saved frame to: {fig_path}")

            if roi_points is not None:
                roi = cv2.bitwise_and(color_img, color_img, mask=mask)

            if include_depth:
                depth_img = depth_videos[cam_name][frame_idx]
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
            
            else:
                # If not including depth, just use the color image
                frame_images.append(color_img)

        # Concatenate all camera pairs horizontally
        final_frame = np.concatenate(frame_images, axis=1)
        out.write(final_frame)   

    out.release()
    print(f'Saved camera videos to: {video_path}')

if __name__ == "__main__":
    # datadir = '/run/user/1000/gvfs/smb-share:server=tos-nas01.local,share=lore_workspace/tests/ballen_2_vaste_locatie/second_test'
    datadir = '../tos_app_data/second_test'
    episode_idx = 129
    main(datadir, episode_idx,
         save_plot=True,
         save_plot_dropoff=True, 
         save_plot_all_dropoff=False, #TODO: debug this, it does not fully work
         save_video=True,
         visualize_robot=True,
         save_robot_video=False #TODO: debug this, it does not fully work
         )
