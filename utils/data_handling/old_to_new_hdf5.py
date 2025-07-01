#!/usr/bin/env python3
"""
Convert old HDF5 dataset format to new format.

Old format:
- action: (N, 7) - robot actions
- observations/action_master: (N, 7) - master/teacher actions  
- observations/action_puppet: (N, 7) - puppet/robot actions
- observations/joint_states: (N, 112) - joint state data
- observations/images/cam_X: (N, H, W, 3) - color images
- observations/depth_images/cam_X: (N, H, W) - depth images

New format:
- robot_positions: joint positions from robot  
- master_positions: joint positions from master
- send_position_robots: commands sent to robot
- robot_position_timestamps: timestamps
- images/cam_X/color: color images
- images/cam_X/depth: depth images
- images/cam_X/color_timestamps: color timestamps
- images/cam_X/depth_timestamps: depth timestamps
- metadata/: episode and camera metadata
"""

import os
import h5py
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
import glob


def convert_old_to_new_hdf5(old_filepath, new_filepath):
    """
    Convert a single old format HDF5 file to new format.
    
    Args:
        old_filepath: Path to old format HDF5 file
        new_filepath: Path to save new format HDF5 file
    """
    print(f"Converting {old_filepath} -> {new_filepath}")
    
    with h5py.File(old_filepath, 'r') as old_f:
        with h5py.File(new_filepath, 'w') as new_f:
            # Extract data from old format
            action = old_f['action'][:]  # (N, 7)
            action_master = old_f['observations']['action_master'][:]  # (N, 7)
            action_puppet = old_f['observations']['action_puppet'][:]  # (N, 7)
            joint_states = old_f['observations']['joint_states'][:]  # (N, 112)
            
            # Get timesteps
            n_timesteps = action.shape[0]
            
            # Create synthetic timestamps (since old format doesn't have them)
            # Assume 20 Hz recording (0.05s interval)
            dt = 0.004
            start_time = 0.0
            timestamps = np.arange(n_timesteps) * dt + start_time
            
            # interpolate action, action_master, action_puppet times 4
            # thus 100 actions become 400 actions
            action = np.repeat(action, 4, axis=0)
            action_master = np.repeat(action_master, 4, axis=0)
            action_puppet = np.repeat(action_puppet, 4, axis=0)

            # Map old data to new format
            # robot_positions = action_puppet (robot's actual positions)
            # master_positions = action_master (human demonstration)
            # send_position_robots = action (commands sent to robot)
            new_f.create_dataset('robot_positions', data=action_puppet.astype(np.float32))
            new_f.create_dataset('master_positions', data=action_master.astype(np.float32))
            new_f.create_dataset('send_position_robots', data=action.astype(np.float32))
            new_f.create_dataset('robot_position_timestamps', data=timestamps.astype(np.float64))
            
            # Convert images
            images_group = new_f.create_group('images')
            
            # Get camera names from old format
            if 'images' in old_f['observations']:
                camera_names = list(old_f['observations']['images'].keys())
            else:
                camera_names = []
            
            for cam_name in camera_names:
                cam_group = images_group.create_group(cam_name)
                
                # Color images
                if cam_name in old_f['observations']['images']:
                    color_images = old_f['observations']['images'][cam_name][:]
                    cam_group.create_dataset('color', data=color_images)
                    
                    # Create synthetic color timestamps
                    color_timestamps = timestamps.copy()  # Same as joint timestamps
                    cam_group.create_dataset('color_timestamps', data=color_timestamps)
                
                # Depth images
                if cam_name in old_f['observations']['depth_images']:
                    depth_images = old_f['observations']['depth_images'][cam_name][:]
                    cam_group.create_dataset('depth', data=depth_images)
                    
                    # Create synthetic depth timestamps
                    depth_timestamps = timestamps.copy()  # Same as joint timestamps
                    cam_group.create_dataset('depth_timestamps', data=depth_timestamps)
            
            # Add metadata
            metadata_group = new_f.create_group('metadata')
            
            # Episode metadata
            episode_idx = int(os.path.basename(old_filepath).split('_')[1].split('.')[0])
            metadata_group.attrs['episode_idx'] = episode_idx
            metadata_group.attrs['start_time'] = start_time
            metadata_group.attrs['end_time'] = start_time + (n_timesteps - 1) * dt
            metadata_group.attrs['record_divisor'] = 4  # Default value
            metadata_group.attrs['total_timesteps'] = n_timesteps
            
            # Camera metadata
            if camera_names:
                camera_metadata = metadata_group.create_group('cameras')
                for cam_name in camera_names:
                    cam_group = camera_metadata.create_group(cam_name)
                    cam_group.attrs['name'] = cam_name
                    cam_group.attrs['serial'] = cam_name
                    
                    # Add image dimensions if available
                    if cam_name in old_f['observations']['images']:
                        color_shape = old_f['observations']['images'][cam_name].shape
                        cam_group.attrs['color_image_shape'] = list(color_shape)
                        cam_group.attrs['color_image_count'] = color_shape[0]
                    
                    if cam_name in old_f['observations']['depth_images']:
                        depth_shape = old_f['observations']['depth_images'][cam_name].shape
                        cam_group.attrs['depth_image_shape'] = list(depth_shape)
                        cam_group.attrs['depth_image_count'] = depth_shape[0]
            
            # Joint state metadata
            joint_metadata = metadata_group.create_group('joint_states')
            joint_metadata.attrs['joint_states_count'] = n_timesteps
            joint_metadata.attrs['fields'] = ['robot_position_timestamps', 'robot_positions', 'master_positions', 'send_position_robots']
            
            # Copy any original attributes
            if hasattr(old_f, 'attrs'):
                for attr_name, attr_value in old_f.attrs.items():
                    metadata_group.attrs[f'original_{attr_name}'] = attr_value


def convert_directory(old_dir, new_dir):
    """
    Convert all HDF5 files in old_dir to new format in new_dir.
    
    Args:
        old_dir: Directory containing old format HDF5 files
        new_dir: Directory to save new format HDF5 files
    """
    # Create output directory
    os.makedirs(new_dir, exist_ok=True)
    
    # Find all HDF5 files in old directory
    old_files = glob.glob(os.path.join(old_dir, "episode_*.hdf5"))
    old_files.sort()
    
    print(f"Found {len(old_files)} HDF5 files to convert")
    
    if not old_files:
        print("No HDF5 files found!")
        return
    
    # Convert each file
    for old_filepath in tqdm(old_files, desc="Converting files"):
        filename = os.path.basename(old_filepath)
        new_filepath = os.path.join(new_dir, filename)
        
        try:
            convert_old_to_new_hdf5(old_filepath, new_filepath)
        except Exception as e:
            print(f"ERROR converting {old_filepath}: {e}")
            continue
    
    print(f"Conversion complete! New files saved in: {new_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert old HDF5 format to new format")
    parser.add_argument("--old_dir", type=str, default="/home/teun/tests/20250522_all_cameras",
                        help="Directory containing old format HDF5 files")
    parser.add_argument("--new_dir", type=str, default=None,
                        help="Directory to save new format HDF5 files (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Generate new directory name with timestamp if not provided
    if args.new_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.new_dir = f"/home/teun/tos_app_data/converted_{timestamp}"
    
    print(f"Converting HDF5 files from {args.old_dir} to {args.new_dir}")
    
    if not os.path.exists(args.old_dir):
        print(f"ERROR: Old directory {args.old_dir} does not exist!")
        return
    
    convert_directory(args.old_dir, args.new_dir)


if __name__ == "__main__":
    main()
