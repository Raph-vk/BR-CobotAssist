#!/usr/bin/env python3
"""
Robot Fanuc Camera Calibration Script

This script provides a simple camera calibration workflow for the Fanuc robot.
It moves the robot to predefined RMI (Robot Machine Interface) positions,
takes pictures at each position, saves the robot state and images for calibration.

Usage:
    python robot_fanuc_cam_calibration.py
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
import cv2
import h5py
import pyrealsense2 as rs
import multiprocessing


# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))


def calcDirectKinematics(jointAngles):
    """
    Forward kinematics for LR Mate 200iD 7L robot.
    
    Args:
        jointAngles (list): The 6 Joint-Angles in degrees [J1, J2, J3, J4, J5, J6]
        
    Returns:
        list: Cartesian position and orientation [x, y, z, w, p, r] in mm and degrees
    """
    def denavitHartenbergMatrix(theta0, jointAngle, d, a, alpha):
        theta = jointAngle + theta0
        matrix = np.array(
            [[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]]
        )
        return matrix
        
    jointAngle1Rad = np.deg2rad(jointAngles[0])
    jointAngle2Rad = np.deg2rad(jointAngles[1])
    jointAngle3Rad = np.deg2rad(jointAngles[2])
    jointAngle4Rad = np.deg2rad(jointAngles[3])
    jointAngle5Rad = np.deg2rad(jointAngles[4])
    jointAngle6Rad = np.deg2rad(jointAngles[5])
    
    matrix1 = denavitHartenbergMatrix(0,
        jointAngle1Rad,
        0,
        50,
        np.pi/2)

    matrix2 = denavitHartenbergMatrix(np.pi/2,
        -jointAngle2Rad,
        0,
        440,
        0)

    matrix3 = denavitHartenbergMatrix(0,
        jointAngle3Rad,
        0,
        35,
        -np.pi/2)

    matrix4 = denavitHartenbergMatrix(0,
        jointAngle4Rad,
        -420,
        0,
        np.pi/2)

    matrix5 = denavitHartenbergMatrix(0,
        jointAngle5Rad,
        0,
        0,
        -np.pi/2)

    matrix6 = denavitHartenbergMatrix(0,
        jointAngle6Rad,
        -80,
        0,
        np.pi)

    matricesMultiplied = matrix1 @ matrix2 @ matrix3 @ matrix4 @ matrix5 @ matrix6

    x = matricesMultiplied[0][3]
    y = matricesMultiplied[1][3]
    z = matricesMultiplied[2][3]
    w = np.rad2deg(np.arctan2(matricesMultiplied[2][1], matricesMultiplied[2][2]))
    try:
        p = np.rad2deg(-np.arcsin(matricesMultiplied[2][0]))
    except RuntimeWarning:
        p = 0.999
        print('Warning, invalid value for the arcsin function')
    r = np.rad2deg(np.arctan2(matricesMultiplied[1][0], matricesMultiplied[0][0]))
    return [x, y, z, w, p, r]


try:
    from utils.utils import setup_logging, load_config
    from modules.robot.fanuc.robot_fanuc import FanucRobot
    from modules.camera.camera_inteld405 import CameraInterfaceIntelD405
except ImportError as e:
    print(f"Warning: Could not import TOS modules: {e}")
    print("Running in standalone mode...")
    
    # Simple logging function for standalone mode
    def setup_logging(component_tag):
        import logging
        logging.basicConfig(level=logging.INFO, format=f'[{component_tag}] %(message)s')
        return logging.getLogger(component_tag)
    
    # Simple config loader for standalone mode
    def load_config(config_path=None):
        return {"hardware": {"robot": {"dof": 6}}}  # Minimal config
    

class FanucCameraCalibration:
    """
    Simple camera calibration script for Fanuc robot.
    Moves robot to RMI positions, captures images and robot states.
    """
    
    def __init__(self, config_path=None):
        """Initialize the calibration system."""
        # Setup logging
        self.logger = setup_logging("FANUC_CAM_CALIBRATION")
        self.logger.info("Initializing Fanuc Camera Calibration...")
        
        # Load configuration
        self.config = load_config(config_path) if config_path else load_config()
        
        # Robot connection parameters
        self.robot = None
        self.cameras = {}  # Dictionary to hold camera instances by serial number
        
        # Target camera serial numbers
        self.target_cameras = ["218622271391", "218622271425"]
        
        # Calibration data storage
        self.calibration_data = {
            "timestamp": datetime.now().isoformat(),
            "robot_positions": [],
            "image_paths": [],
            "robot_states": [],
            "camera_data": {}
        }
        
        # Create output directory for calibration data in tos_app_data/camera_calibration
        # Get the data directory from config
        try:
            from utils.utils import load_config
            config = load_config()
            data_directory = config.get("general", {}).get("data_directory", "../tos_app_data")
        except ImportError:
            data_directory = "../tos_app_data"  # Fallback if utils not available
        
        # Convert relative path to absolute based on project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "../../../../"))
        data_path = os.path.join(project_root, data_directory.lstrip("../"))
        
        self.output_dir = os.path.join(data_path, "camera_calibration", "multi_pos_calibration", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Calibration data will be saved to: {self.output_dir}")
        
        self.rmi_positions = self._define_rmi_positions()
        
    def _define_rmi_positions(self):
        """
        Define the RMI (Robot Machine Interface) positions for calibration.
               
        Returns:
            list: List of joint positions [J1, J2, J3, J4, J5, J6, gripper_state]
        """
        # 50 calibration positions with base position [0, 0, 0, 0, -90, 180] (straight pose with J5 at -90° and J6 at 180°)
        # All joints can vary by maximum ±5 degrees from base position
        positions = [
            # Base position and single joint variations
            [0, 0, 0, 0, -90, 180, 0],         # Position 1: Base reference position
            [5, 0, 0, 0, -90, 180, 0],         # Position 2: J1 +5°
            [-5, 0, 0, 0, -90, 180, 0],        # Position 3: J1 -5°
            [0, 5, 0, 0, -90, 180, 0],         # Position 4: J2 +5°
            [0, -5, 0, 0, -90, 180, 0],        # Position 5: J2 -5°
            [0, 0, 5, 0, -90, 180, 0],         # Position 6: J3 +5°
            [0, 0, -5, 0, -90, 180, 0],        # Position 7: J3 -5°
            [0, 0, 0, 5, -90, 180, 0],         # Position 8: J4 +5°
            [0, 0, 0, -5, -90, 180, 0],        # Position 9: J4 -5°
            [0, 0, 0, 0, -85, 180, 0],         # Position 10: J5 +5° (-85°)
            [0, 0, 0, 0, -95, 180, 0],         # Position 11: J5 -5° (-95°)
            [0, 0, 0, 0, -90, 185, 0],         # Position 12: J6 +5°
            [0, 0, 0, 0, -90, 175, 0],         # Position 13: J6 -5°
            
            # Small combined variations (±1 to ±2 degrees)
            [1, 1, 1, 1, -89, 181, 0],         # Position 14: Small positive offsets
            [-1, -1, -1, -1, -91, 179, 0],     # Position 15: Small negative offsets
            [2, 1, -1, 2, -88, 178, 0],        # Position 16: Mixed small offsets
            [-2, -1, 1, -2, -92, 182, 0],      # Position 17: Mixed small offsets
            [1, -2, 2, -1, -89, 181, 0],       # Position 18: Mixed variations
            [-1, 2, -2, 1, -91, 179, 0],       # Position 19: Mixed variations
            [2, 2, 0, 0, -88, 180, 0],         # Position 20: Partial variations
            
            # Medium combined variations (±3 degrees)
            [3, 3, 3, 3, -87, 183, 0],         # Position 21: Medium positive
            [-3, -3, -3, -3, -93, 177, 0],     # Position 22: Medium negative
            [3, -3, 3, -3, -87, 177, 0],       # Position 23: Alternating signs
            [-3, 3, -3, 3, -93, 183, 0],       # Position 24: Alternating signs
            [2, -3, 1, 3, -88, 179, 0],        # Position 25: Random medium mix
            [-2, 3, -1, -3, -92, 181, 0],      # Position 26: Random medium mix
            [3, 1, -3, 2, -87, 178, 0],        # Position 27: Varied medium
            [-3, -1, 3, -2, -93, 182, 0],      # Position 28: Varied medium
            [1, 3, -2, 3, -89, 177, 0],        # Position 29: Mixed medium
            [-1, -3, 2, -3, -91, 183, 0],      # Position 30: Mixed medium
            
            # Large combined variations (±4 to ±5 degrees)
            [4, 4, 4, 4, -86, 184, 0],         # Position 31: Large positive
            [-4, -4, -4, -4, -94, 176, 0],     # Position 32: Large negative
            [5, 5, 5, 5, -85, 185, 0],         # Position 33: Maximum positive
            [-5, -5, -5, -5, -95, 175, 0],     # Position 34: Maximum negative
            [5, -5, 5, -5, -85, 175, 0],       # Position 35: Max alternating
            [-5, 5, -5, 5, -95, 185, 0],       # Position 36: Max alternating
            [4, -5, 3, 5, -86, 176, 0],        # Position 37: Random large mix
            [-4, 5, -3, -5, -94, 184, 0],      # Position 38: Random large mix
            [5, 2, -4, 3, -85, 179, 0],        # Position 39: Varied large
            [-5, -2, 4, -3, -95, 181, 0],      # Position 40: Varied large
            
            # Additional mixed patterns
            [3, 0, -4, 2, -87, 175, 0],        # Position 41: Sparse pattern
            [-3, 0, 4, -2, -93, 185, 0],       # Position 42: Sparse pattern
            [0, 4, 0, -3, -90, 182, 0],        # Position 43: Selective joints
            [0, -4, 0, 3, -90, 178, 0],        # Position 44: Selective joints
            [2, -1, 5, 0, -88, 177, 0],        # Position 45: Asymmetric
            [-2, 1, -5, 0, -92, 183, 0],       # Position 46: Asymmetric
            [4, 3, -2, -4, -86, 181, 0],       # Position 47: Complex mix
            [-4, -3, 2, 4, -94, 179, 0],       # Position 48: Complex mix
            [1, 5, -3, 4, -89, 176, 0],        # Position 49: Final variation
            [-1, -5, 3, -4, -91, 184, 0],      # Position 50: Final variation
        ]

        # create an offset for each position
        offset = [0, 0, 0, 0, 0, 0, 0]  # No gripper state for calibration
        positions = [list(np.array(pos) + np.array(offset)) for pos in positions]
        # Convert NumPy values back to Python floats to avoid JSON serialization issues
        positions = [[float(val) for val in pos] for pos in positions]
        
        self.logger.info(f"Defined {len(positions)} RMI calibration positions")
        return positions
    
    def connect_robot(self):
        """Connect to the Fanuc robot."""
        try:
            self.logger.info("Connecting to Fanuc robot...")
            
            # Create dummy queues for robot interface (calibration doesn't need full interface)
            robot_interface_commup = multiprocessing.Queue()
            shm_target_pos1 = multiprocessing.Queue()
            shm_joint_data1 = multiprocessing.Queue()
            shm_joint_data2 = multiprocessing.Queue()
            
            # Initialize robot
            # TODO: Adjust shm_target_pos2_info based on your system setup
            shm_target_pos2_info = None  # Placeholder
            
            self.robot = FanucRobot(
                robot_interface_commup=robot_interface_commup,
                shm_target_pos1=shm_target_pos1,
                shm_target_pos2_info=shm_target_pos2_info,
                shm_joint_data1=shm_joint_data1,
                shm_joint_data2=shm_joint_data2,
                logger_ri=self.logger,
                config=self.config
            )
            
            self.logger.info("Robot connected successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to robot: {e}")
            return False
    
    def connect_camera(self):
        """
        Connect to the D405 cameras with highest resolution.
        """
        try:
            self.logger.info("Connecting to Intel D405 cameras...")
            
            # Discover available cameras
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                self.logger.error("No RealSense devices found")
                return False
            
            # Find and initialize target cameras
            for device in devices:
                serial_number = device.get_info(rs.camera_info.serial_number)
                device_name = device.get_info(rs.camera_info.name)
                
                if serial_number in self.target_cameras:
                    self.logger.info(f"Found target camera: {device_name} (SN: {serial_number})")
                    
                    # Get highest resolution for this camera (color only)
                    color_res = self._get_highest_resolution(device)
                    
                    # Initialize camera with highest resolution (color only)
                    camera_instance = self._initialize_d405_camera(serial_number, color_res)
                    
                    if camera_instance:
                        self.cameras[serial_number] = camera_instance
                        self.logger.info(f"Camera {serial_number} initialized with color resolution: {color_res}")
                    else:
                        self.logger.error(f"Failed to initialize camera {serial_number}")
                        return False
            
            # Check if we found all target cameras
            if len(self.cameras) != len(self.target_cameras):
                missing = set(self.target_cameras) - set(self.cameras.keys())
                self.logger.error(f"Missing target cameras: {missing}")
                return False
            
            self.logger.info(f"Successfully connected to {len(self.cameras)} cameras")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to cameras: {e}")
            return False
    
    def _get_highest_resolution(self, device):
        """
        Get the highest available resolution for color stream only.
        """
        try:
            color_resolutions = []
            
            # Query available color stream profiles only
            for sensor in device.query_sensors():
                if sensor.is_color_sensor():
                    for profile in sensor.get_stream_profiles():
                        if profile.stream_type() == rs.stream.color:
                            vp = profile.as_video_stream_profile()
                            color_resolutions.append((vp.width(), vp.height()))
            
            # Remove duplicates and sort by total pixels (width * height)
            color_resolutions = sorted(list(set(color_resolutions)), 
                                     key=lambda x: x[0] * x[1], reverse=True)
            
            # Return highest resolution
            color_res = color_resolutions[0] if color_resolutions else (1280, 720)
            
            self.logger.info(f"Available color resolutions: {color_resolutions[:3]}...")  # Show top 3
            self.logger.info(f"Selected color resolution: {color_res}")
            
            return color_res
            
        except Exception as e:
            self.logger.error(f"Error getting camera resolutions: {e}")
            return (640, 480)  # Fallback resolution
    
    def _initialize_d405_camera(self, serial_number, color_res):
        """
        Initialize a single D405 camera with specified color resolution only.
        """
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            
            # Configure camera - COLOR ONLY
            config.enable_device(serial_number)
            config.enable_stream(rs.stream.color, color_res[0], color_res[1], 
                               rs.format.rgb8, 30)  # Frame rate doesn't matter per requirement
            
            # Start pipeline
            pipeline_profile = pipeline.start(config)
            device = pipeline_profile.get_device()
            
            # Enable global time if supported
            sensors = device.query_sensors()
            for sensor in sensors:
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
            
            # Wait for camera to stabilize
            for _ in range(30):  # Skip first 30 frames
                pipeline.wait_for_frames()
            
            camera_info = {
                'pipeline': pipeline,
                'config': config,
                'device': device,
                'color_resolution': color_res,
                'serial_number': serial_number
            }
            
            return camera_info
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera {serial_number}: {e}")
            return None
    
    def move_to_rmi_position(self, position_index):
        """
        Move robot to the specified RMI position.
        
        Args:
            position_index (int): Index of the RMI position to move to
            
        Returns:
            bool: True if movement successful, False otherwise
        """
        if position_index >= len(self.rmi_positions):
            self.logger.error(f"Invalid position index: {position_index}")
            return False
        
        position = self.rmi_positions[position_index]
        self.logger.info(f"Moving to RMI position {position_index + 1}: {position}")
        
        try:
            # Move robot to position
            success = self.robot.push_joint_motion(
                position=position[:6],  # First 6 values are joint positions
                speed=10,
                term_type="FINE",
                term_val=0
            )
            
            if success:
                self.logger.info(f"Successfully moved to position {position_index + 1}")
                # Wait for robot to settle
                time.sleep(2.0)
                return True
            else:
                self.logger.error(f"Failed to move to position {position_index + 1}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error moving to position {position_index + 1}: {e}")
            return False
    
    def wait_for_confirmation(self, position_index):
        """
        DEPRECATED: This function is no longer used in automatic mode.
        Originally waited for user confirmation that the robot is in the correct position.
        
        Args:
            position_index (int): Current position index
            
        Returns:
            bool: Always returns True in automatic mode
        """
        # Automatic mode - no user confirmation needed
        print(f"\n--- RMI Position {position_index + 1} ---")
        print(f"Robot moved to position: {self.rmi_positions[position_index]}")
        print("Proceeding automatically with image capture...")
        return True
    
    def capture_image(self, position_index):
        """
        Capture images from all connected D405 cameras.
        
        Args:
            position_index (int): Current position index
            
        Returns:
            dict: Dictionary with captured image data for each camera
        """
        try:
            self.logger.info(f"Capturing images from {len(self.cameras)} cameras at position {position_index + 1}")
            
            captured_data = {}
            
            for serial_number, camera_info in self.cameras.items():
                try:
                    # Get frames from camera
                    frames = camera_info['pipeline'].wait_for_frames(timeout_ms=5000)
                    
                    color_frame = frames.get_color_frame()
                    
                    if not color_frame:
                        self.logger.error(f"Failed to get color frame from camera {serial_number}")
                        continue
                    
                    # Convert to numpy array
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # Get timestamp
                    color_timestamp = color_frame.get_timestamp() / 1000.0
                    
                    # Store captured data
                    captured_data[serial_number] = {
                        'color_image': color_image,
                        'color_timestamp': color_timestamp,
                        'color_resolution': camera_info['color_resolution']
                    }
                    
                    self.logger.info(f"Captured color image from camera {serial_number}: "
                                   f"Shape {color_image.shape}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to capture from camera {serial_number}: {e}")
                    continue
            
            if not captured_data:
                self.logger.error("No images captured from any camera")
                return None
            
            return captured_data
            
        except Exception as e:
            self.logger.error(f"Failed to capture images at position {position_index + 1}: {e}")
            return None
    
    def save_robot_state(self, position_index):
        """
        Save current robot state (joint positions, pose, etc.).
        
        Args:
            position_index (int): Current position index
            
        Returns:
            dict: Robot state data
        """
        try:
            self.logger.info(f"Saving robot state at position {position_index + 1}")
            
            # Get current robot position
            # TODO: Implement actual robot state reading
            current_position = self.rmi_positions[position_index]
            joint_positions = current_position[:6]
            
            # Calculate cartesian position using forward kinematics
            cartesian_position = calcDirectKinematics(joint_positions)
            
            robot_state = {
                "position_index": position_index + 1,
                "joint_positions": joint_positions,
                "cartesian_position": {
                    "x": cartesian_position[0],
                    "y": cartesian_position[1], 
                    "z": cartesian_position[2],
                    "w": cartesian_position[3],  # Roll (W)
                    "p": cartesian_position[4],  # Pitch (P)
                    "r": cartesian_position[5]   # Yaw (R)
                },
                "gripper_state": current_position[6],
                "timestamp": datetime.now().isoformat(),
                # TODO: Add more robot state information as needed
                # "tcp_pose": self.robot.get_tcp_pose(),  # Tool center point pose
                # "joint_velocities": self.robot.get_joint_velocities(),
                # "joint_torques": self.robot.get_joint_torques(),
            }
            
            self.logger.info(f"Robot state saved for position {position_index + 1}")
            self.logger.info(f"Joint positions: {joint_positions}")
            self.logger.info(f"Cartesian position: X={cartesian_position[0]:.2f}, Y={cartesian_position[1]:.2f}, Z={cartesian_position[2]:.2f}")
            self.logger.info(f"Orientation: W={cartesian_position[3]:.2f}, P={cartesian_position[4]:.2f}, R={cartesian_position[5]:.2f}")
            
            return robot_state
            
        except Exception as e:
            self.logger.error(f"Failed to save robot state at position {position_index + 1}: {e}")
            return None
    
    def save_hdf5_data(self, position_index, robot_state, captured_images):
        """
        Save robot state and camera images to HDF5 file for one position.
        
        Args:
            position_index (int): Current position index
            robot_state (dict): Robot state data
            captured_images (dict): Captured image data from cameras
            
        Returns:
            str: Path to saved HDF5 file
        """
        try:
            hdf5_filename = f"calibration_pos_{position_index + 1:02d}.h5"
            hdf5_path = os.path.join(self.output_dir, hdf5_filename)
            
            self.logger.info(f"Saving HDF5 data to: {hdf5_path}")
            
            with h5py.File(hdf5_path, 'w') as f:
                # Save robot state
                robot_group = f.create_group('robot_state')
                robot_group.attrs['position_index'] = position_index + 1
                robot_group.attrs['timestamp'] = robot_state['timestamp']
                robot_group.create_dataset('joint_positions', data=np.array(robot_state['joint_positions']))
                robot_group.attrs['gripper_state'] = robot_state['gripper_state']
                
                # Save cartesian position data
                cartesian_group = robot_group.create_group('cartesian_position')
                cartesian_data = robot_state['cartesian_position']
                cartesian_group.create_dataset('position', data=np.array([
                    cartesian_data['x'], cartesian_data['y'], cartesian_data['z']
                ]))
                cartesian_group.create_dataset('orientation', data=np.array([
                    cartesian_data['w'], cartesian_data['p'], cartesian_data['r']
                ]))
                cartesian_group.attrs['x'] = cartesian_data['x']
                cartesian_group.attrs['y'] = cartesian_data['y']
                cartesian_group.attrs['z'] = cartesian_data['z']
                cartesian_group.attrs['w'] = cartesian_data['w']  # Roll
                cartesian_group.attrs['p'] = cartesian_data['p']  # Pitch
                cartesian_group.attrs['r'] = cartesian_data['r']  # Yaw
                
                # Save camera data
                cameras_group = f.create_group('cameras')
                
                for serial_number, image_data in captured_images.items():
                    camera_group = cameras_group.create_group(f'camera_{serial_number}')
                    
                    # Save camera attributes
                    camera_group.attrs['serial_number'] = serial_number
                    camera_group.attrs['color_timestamp'] = image_data['color_timestamp']
                    camera_group.attrs['color_resolution'] = image_data['color_resolution']
                    
                    # Save color image only
                    camera_group.create_dataset('color_image', 
                                              data=image_data['color_image'],
                                              compression='gzip', 
                                              compression_opts=6)
                    
                    self.logger.info(f"Saved color image for camera {serial_number} to HDF5")
                
                # Add metadata
                f.attrs['created_timestamp'] = datetime.now().isoformat()
                f.attrs['position_index'] = position_index + 1
                f.attrs['target_joint_position'] = self.rmi_positions[position_index]
                f.attrs['num_cameras'] = len(captured_images)
                f.attrs['camera_serial_numbers'] = list(captured_images.keys())
            
            self.logger.info(f"HDF5 file saved successfully: {hdf5_path}")
            return hdf5_path
            
        except Exception as e:
            self.logger.error(f"Failed to save HDF5 data: {e}")
            return None
    
    def save_calibration_data(self):
        """Save all calibration data to JSON file."""
        try:
            calibration_file = os.path.join(self.output_dir, "calibration_data.json")
            
            with open(calibration_file, 'w') as f:
                json.dump(self.calibration_data, f, indent=4)
            
            self.logger.info(f"Calibration data saved to: {calibration_file}")
            return calibration_file
            
        except Exception as e:
            self.logger.error(f"Failed to save calibration data: {e}")
            return None
    
    def run_calibration(self):
        """
        Main calibration workflow.
        Moves through all RMI positions, captures images and robot states.
        """
        self.logger.info("Starting camera calibration workflow...")
        
        # Connect to robot and camera
        if not self.connect_robot():
            self.logger.error("Failed to connect to robot. Exiting.")
            return False
        
        if not self.connect_camera():
            self.logger.error("Failed to connect to camera. Exiting.")
            return False
        
        print(f"\n=== Fanuc Robot Camera Calibration ===")
        print(f"Target cameras: {self.target_cameras}")
        print(f"Total RMI positions to visit: {len(self.rmi_positions)}")
        print(f"Output directory: {self.output_dir}")
        print("Mode: AUTOMATIC - No user confirmation required")
        print("Delay between movement and capture: 0.1 seconds")
        print("=" * 50)
        
        # Process each RMI position automatically
        for position_index in range(len(self.rmi_positions)):
            self.logger.info(f"Processing position {position_index + 1}/{len(self.rmi_positions)}")
            
            # Move to RMI position
            if not self.move_to_rmi_position(position_index):
                self.logger.error(f"Failed to move to position {position_index + 1}, skipping...")
                continue
            
            # Small delay to ensure robot has settled before taking picture
            self.logger.info(f"Robot moved to position {position_index + 1}, waiting 0.1s before capture...")
            time.sleep(1)
            
            # Capture images from all cameras
            captured_images = self.capture_image(position_index)
            if captured_images is None:
                self.logger.error(f"Failed to capture images at position {position_index + 1}")
                continue
            
            # Save robot state
            robot_state = self.save_robot_state(position_index)
            if robot_state is None:
                self.logger.error(f"Failed to save robot state at position {position_index + 1}")
                continue
            
            # Save data to HDF5 file
            hdf5_path = self.save_hdf5_data(position_index, robot_state, captured_images)
            if hdf5_path is None:
                self.logger.error(f"Failed to save HDF5 data at position {position_index + 1}")
                continue
            
            # Store summary data for JSON output
            self.calibration_data["robot_positions"].append(self.rmi_positions[position_index])
            self.calibration_data["robot_states"].append(robot_state)
            self.calibration_data["camera_data"][f"position_{position_index + 1}"] = {
                "hdf5_file": hdf5_path,
                "cameras": list(captured_images.keys()),
                "timestamp": robot_state['timestamp']
            }
            
            self.logger.info(f"Completed position {position_index + 1}")
            print(f"Position {position_index + 1} completed successfully!")
            print(f"HDF5 file: {hdf5_path}")
            print(f"Cameras captured: {list(captured_images.keys())}")
            
            # Automatically continue to next position
            if position_index < len(self.rmi_positions) - 1:  # Not the last position
                print(f"Automatically proceeding to position {position_index + 2}...")
            
            print()  # Empty line for readability
        
        # Save all calibration data
        calibration_file = self.save_calibration_data()
        
        print(f"\n=== Calibration Complete ===")
        print(f"Captured data from {len(self.calibration_data['camera_data'])} positions")
        print(f"Target cameras: {self.target_cameras}")
        print(f"Data saved to: {self.output_dir}")
        print(f"Calibration file: {calibration_file}")
        
        self.logger.info("Camera calibration workflow completed")
        return True
    
    def cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up calibration system...")
        
        # Cleanup robot connection
        if self.robot:
            try:
                self.robot.stop()
                self.robot._cleanup_connection()
            except Exception as e:
                self.logger.error(f"Error cleaning up robot: {e}")
        
        # Cleanup camera connections
        if self.cameras:
            try:
                for serial_number, camera_info in self.cameras.items():
                    try:
                        camera_info['pipeline'].stop()
                        self.logger.info(f"Stopped camera {serial_number}")
                    except Exception as e:
                        self.logger.error(f"Error stopping camera {serial_number}: {e}")
                self.cameras.clear()
            except Exception as e:
                self.logger.error(f"Error cleaning up cameras: {e}")
        
        self.logger.info("Cleanup completed")


def main():
    """Main entry point for the calibration script."""
    calibration = None
    
    try:
        # Initialize calibration system
        calibration = FanucCameraCalibration()
        
        # Run calibration workflow
        success = calibration.run_calibration()
        
        if success:
            print("Calibration completed successfully!")
        else:
            print("Calibration failed or was interrupted.")
    
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user.")
    
    except Exception as e:
        print(f"Calibration error: {e}")
    
    finally:
        # Cleanup
        if calibration:
            calibration.cleanup()


if __name__ == "__main__":
    main()
