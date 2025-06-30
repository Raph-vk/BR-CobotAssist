import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

def inspect_calibration(calibration_type="single_pos"):
    """
    Inspects calibration data by finding the latest created folder in tos_app_data/camera_calibration,
    selecting a random image from the HDF5 files, and visualizing it with imshow.
    Also prints the data format information.
    
    Args:
        calibration_type (str): Type of calibration to inspect. Options:
                              "single_pos" for single position calibration
                              "multi_pos" for multi-position calibration
    """
    
    # Define the calibration directory path in tos_app_data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../../../"))
    
    if calibration_type == "multi_pos":
        calibration_dir = os.path.join(project_root, "tos_app_data", "camera_calibration", "multi_pos_calibration")
    else:
        calibration_dir = os.path.join(project_root, "tos_app_data", "camera_calibration", "single_pos_calibration")
    
    # Check if calibration directory exists
    if not os.path.exists(calibration_dir):
        print(f"Calibration directory does not exist: {calibration_dir}")
        print("Make sure you have run the calibration script at least once.")
        return
    
    # Get all subdirectories (excluding files)
    folders = [d for d in os.listdir(calibration_dir) 
               if os.path.isdir(os.path.join(calibration_dir, d))]
    
    if not folders:
        print("No calibration folders found!")
        return
    
    # Sort folders by name (assuming timestamp format YYYYMMDD_HHMMSS)
    folders.sort(reverse=True)  # Latest first
    latest_folder = folders[0]
    
    print(f"Latest calibration folder: {latest_folder}")
    
    # Path to the latest calibration folder
    latest_path = os.path.join(calibration_dir, latest_folder)
    
    # Find all HDF5 files in the folder
    h5_files = [f for f in os.listdir(latest_path) if f.endswith('.h5')]
    
    if not h5_files:
        print("No HDF5 files found in the latest calibration folder!")
        return
    
    print(f"Found {len(h5_files)} HDF5 files: {h5_files}")
    
    # Select a random HDF5 file
    selected_file = random.choice(h5_files)
    file_path = os.path.join(latest_path, selected_file)
    
    print(f"Selected file: {selected_file}")
    
    # Open and inspect the HDF5 file
    with h5py.File(file_path, 'r') as f:
        print("\n=== HDF5 File Structure ===")
        
        def print_structure(name, obj):
            print(f"{name}: {type(obj)}")
            if isinstance(obj, h5py.Dataset):
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                print(f"  Min: {np.min(obj[:])}, Max: {np.max(obj[:])}")
        
        f.visititems(print_structure)
        
        # Display robot state information
        if 'robot_state' in f:
            print("\n=== Robot State Information ===")
            robot_group = f['robot_state']
            
            # Joint positions
            if 'joint_positions' in robot_group:
                joint_pos = robot_group['joint_positions'][:]
                print(f"Joint positions: {joint_pos}")
            
            # Cartesian position (if available)
            if 'cartesian_position' in robot_group:
                cart_group = robot_group['cartesian_position']
                print(f"Cartesian Position:")
                if 'position' in cart_group:
                    pos = cart_group['position'][:]
                    print(f"  Position (x, y, z): {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f} mm")
                if 'orientation' in cart_group:
                    orient = cart_group['orientation'][:]
                    print(f"  Orientation (w, p, r): {orient[0]:.2f}, {orient[1]:.2f}, {orient[2]:.2f} degrees")
                
                # Also print individual attributes if available
                if hasattr(cart_group, 'attrs'):
                    attrs = dict(cart_group.attrs)
                    if 'x' in attrs:
                        print(f"  X: {attrs['x']:.2f} mm")
                        print(f"  Y: {attrs['y']:.2f} mm") 
                        print(f"  Z: {attrs['z']:.2f} mm")
                        print(f"  W (Roll): {attrs['w']:.2f} degrees")
                        print(f"  P (Pitch): {attrs['p']:.2f} degrees")
                        print(f"  R (Yaw): {attrs['r']:.2f} degrees")
            
            # Other robot attributes
            if hasattr(robot_group, 'attrs'):
                attrs = dict(robot_group.attrs)
                if 'gripper_state' in attrs:
                    print(f"Gripper state: {attrs['gripper_state']}")
                if 'timestamp' in attrs:
                    print(f"Timestamp: {attrs['timestamp']}")
                if 'position_index' in attrs:
                    print(f"Position index: {attrs['position_index']}")
        
        # Look for image data in cameras group and other locations
        image_keys = []
        
        # Check for cameras group structure
        if 'cameras' in f:
            cameras_group = f['cameras']
            for camera_id in cameras_group.keys():
                camera_group = cameras_group[camera_id]
                if 'color_image' in camera_group:
                    dataset_path = f"cameras/{camera_id}/color_image"
                    image_keys.append(dataset_path)
        
        # Also check for other image datasets at root level
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                # Check if this looks like image data (2D or 3D array)
                if len(f[key].shape) >= 2 and 'image' in key.lower():
                    image_keys.append(key)
        
        if not image_keys:
            print("No image datasets found in the HDF5 file!")
            return
        
        print(f"\nFound potential image datasets: {image_keys}")
        
        # Select the first image dataset or a random one if multiple
        selected_key = image_keys[0] if len(image_keys) == 1 else random.choice(image_keys)
        print(f"Visualizing dataset: {selected_key}")
        
        # Load the image data (handle nested structure)
        if '/' in selected_key:
            # Navigate through the nested structure
            parts = selected_key.split('/')
            current = f
            for part in parts:
                current = current[part]
            image_data = current[:]
        else:
            # Direct access
            image_data = f[selected_key][:]
        
        print(f"\n=== Selected Image Data Format ===")
        print(f"Dataset key: {selected_key}")
        print(f"Shape: {image_data.shape}")
        print(f"Data type: {image_data.dtype}")
        print(f"Min value: {np.min(image_data)}")
        print(f"Max value: {np.max(image_data)}")
        print(f"Mean value: {np.mean(image_data):.2f}")
        print(f"Standard deviation: {np.std(image_data):.2f}")
        
        # Handle different image formats
        if len(image_data.shape) == 4:  # Multiple images
            # Select a random image from the batch
            random_idx = random.randint(0, image_data.shape[0] - 1)
            image_to_show = image_data[random_idx]
            print(f"Selected image index {random_idx} from batch of {image_data.shape[0]} images")
        elif len(image_data.shape) == 3:
            if image_data.shape[2] in [1, 3, 4]:  # Single image with channels
                image_to_show = image_data
            else:  # Multiple grayscale images or different format
                # Take the first image
                image_to_show = image_data[0] if image_data.shape[0] < image_data.shape[2] else image_data[:, :, 0]
        else:  # 2D grayscale image
            image_to_show = image_data
        
        # Visualize the image
        plt.figure(figsize=(10, 8))
        
        if len(image_to_show.shape) == 3 and image_to_show.shape[2] == 1:
            # Remove single channel dimension for grayscale
            image_to_show = image_to_show.squeeze()
        
        if len(image_to_show.shape) == 2:  # Grayscale
            plt.imshow(image_to_show, cmap='gray')
            plt.colorbar(label='Intensity')
        else:  # Color image
            plt.imshow(image_to_show)
        
        plt.title(f'Calibration Image from {selected_file}\nDataset: {selected_key}\nShape: {image_to_show.shape}')
        plt.axis('on')  # Show axes to see pixel coordinates
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import sys
    
    # Check if calibration type is provided as command line argument
    if len(sys.argv) > 1:
        calibration_type = sys.argv[1]
        if calibration_type not in ["single_pos", "multi_pos"]:
            print("Invalid calibration type. Use 'single_pos' or 'multi_pos'")
            print("Usage: python inspect_calibration_hdf5.py [single_pos|multi_pos]")
            sys.exit(1)
    else:
        calibration_type = "single_pos"  # Default to single position
    
    print(f"Inspecting {calibration_type} calibration data...")
    inspect_calibration(calibration_type)