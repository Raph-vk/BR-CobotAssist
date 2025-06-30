#!/usr/bin/env python3
"""
Robot Fanuc Single Position Camera Calibration Script

This script moves the robot to a single specified position and takes 50 photos,
each time after user confirmation via keyboard press. Saves robot state and images.

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
    
# Platform-specific imports
if sys.platform == "win32":
    import msvcrt
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))



    

class FanucSinglePositionCalibration:
    """
    Single position camera calibration script for Fanuc robot.
    Moves robot to one position and takes multiple photos with user confirmation.
    """
    
    def __init__(self, config_path=None):
        """Initialize the calibration system."""
        # Setup logging
        self.logger = setup_logging("FANUC_SINGLE_POS_CALIBRATION")
        self.logger.info("Initializing Fanuc Single Position Camera Calibration...")
        
        # Load configuration
        self.config = load_config(config_path) if config_path else load_config()
        
        # Robot connection parameters
        self.robot = None
        self.cameras = {}  # Dictionary to hold camera instances by serial number
        
        # Target camera serial numbers (matching config.yaml)
        # self.target_cameras = ["218622271391", "218622271425", "130322272626"] # Run1
        self.target_cameras = ["218622271391", "218622271425", "218622271529"]  # Run2
        
        # Calibration parameters
        self.num_photos = 50
        self.target_position = [0, 0, 0, 0, -90, 180, 0]  # Default position
        
        # Calibration data storage
        self.calibration_data = {
            "timestamp": datetime.now().isoformat(),
            "target_position": self.target_position,
            "num_photos": self.num_photos,
            "photos_taken": [],
            "robot_states": [],
            "camera_data": {}
        }
        
        # Create output directory for calibration data in tos_app_data/camera_calibration
        # Get the data directory from config
        config = load_config()
        data_directory = config.get("general", {}).get("data_directory", "../tos_app_data")
        
        # Convert relative path to absolute based on project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "../../../../"))
        data_path = os.path.join(project_root, data_directory.lstrip("../"))
        
        self.output_dir = os.path.join(data_path, "camera_calibration", "single_pos_calibration", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Calibration data will be saved to: {self.output_dir}")
        
    def set_target_position(self, position):
        """
        Set the target position for calibration.
        
        Args:
            position (list): Joint positions [J1, J2, J3, J4, J5, J6, gripper_state]
        """
        self.target_position = position
        self.calibration_data["target_position"] = position
        self.logger.info(f"Target position set to: {position}")
        
    def connect_robot(self):
        """Connect to the Fanuc robot."""
        try:
            self.logger.info("Connecting to Fanuc robot...")
            
            # Create dummy queues for robot interface
            robot_interface_commup = multiprocessing.Queue()
            shm_target_pos1 = multiprocessing.Queue()
            shm_joint_data1 = multiprocessing.Queue()
            shm_joint_data2 = multiprocessing.Queue()
            
            # Initialize robot
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
        """Connect to Intel D405 cameras."""
        try:
            self.logger.info("Connecting to cameras...")
            
            # Initialize RealSense context
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                self.logger.error("No RealSense devices found")
                return False
            
            # Connect to target cameras
            for device in devices:
                serial_number = device.get_info(rs.camera_info.serial_number)
                
                if serial_number in self.target_cameras:
                    self.logger.info(f"Found target camera: {serial_number}")
                    
                    # Create pipeline for this camera
                    pipeline = rs.pipeline()
                    config = rs.config()
                    config.enable_device(serial_number)
                    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
                    
                    # Start streaming
                    pipeline.start(config)
                    
                    self.cameras[serial_number] = {
                        'pipeline': pipeline,
                        'config': config
                    }
                    
                    self.logger.info(f"Camera {serial_number} connected successfully")
            
            if len(self.cameras) == 0:
                self.logger.error("No target cameras connected")
                return False
            
            self.logger.info(f"Connected to {len(self.cameras)} cameras")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to cameras: {e}")
            return False
    
    def move_to_target_position(self):
        """Move robot to the target position."""
        self.logger.info(f"Moving to target position: {self.target_position}")
        
        try:
            # Move robot to position
            success = self.robot.push_joint_motion(
                position=self.target_position[:6],  # First 6 values are joint positions
                speed=10,
                term_type="FINE",
                term_val=0
            )
            
            if success:
                self.logger.info("Successfully moved to target position")
                # Wait for robot to settle
                time.sleep(2.0)
                return True
            else:
                self.logger.error("Failed to move to target position")
                return False
                
        except Exception as e:
            self.logger.error(f"Error moving to target position: {e}")
            return False
    
    def capture_image(self, photo_index):
        """
        Capture images from all connected cameras.
        
        Args:
            photo_index (int): Current photo index
            
        Returns:
            dict: Dictionary with captured image data for each camera
        """
        try:
            self.logger.info(f"Capturing photo {photo_index + 1}/{self.num_photos}")
            
            captured_data = {}
            
            for serial_number, camera_info in self.cameras.items():
                try:
                    # Wait for frames
                    frames = camera_info['pipeline'].wait_for_frames(timeout_ms=5000)
                    color_frame = frames.get_color_frame()
                    
                    if not color_frame:
                        self.logger.error(f"No color frame from camera {serial_number}")
                        continue
                    
                    # Convert to numpy array
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # Store captured data
                    captured_data[serial_number] = {
                        'color_image': color_image,
                        'color_timestamp': time.time(),
                        'color_resolution': (color_image.shape[1], color_image.shape[0])
                    }
                    
                    self.logger.info(f"Captured image from camera {serial_number}: {color_image.shape}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to capture from camera {serial_number}: {e}")
                    continue
            
            return captured_data if captured_data else None
            
        except Exception as e:
            self.logger.error(f"Failed to capture images for photo {photo_index + 1}: {e}")
            return None
    
    def save_robot_state(self, photo_index):
        """
        Save current robot state (joint positions, pose, etc.).
        
        Args:
            photo_index (int): Current photo index
            
        Returns:
            dict: Robot state data
        """
        try:
            self.logger.info(f"Saving robot state for photo {photo_index + 1}")
            
            # Get current robot position
            joint_positions = self.target_position[:6]
            
            # Calculate cartesian position using forward kinematics
            cartesian_position = calcDirectKinematics(joint_positions)
            
            robot_state = {
                "photo_index": photo_index + 1,
                "joint_positions": joint_positions,
                "cartesian_position": {
                    "x": cartesian_position[0],
                    "y": cartesian_position[1], 
                    "z": cartesian_position[2],
                    "w": cartesian_position[3],  # Roll (W)
                    "p": cartesian_position[4],  # Pitch (P)
                    "r": cartesian_position[5]   # Yaw (R)
                },
                "gripper_state": self.target_position[6],
                "timestamp": datetime.now().isoformat(),
            }
            
            self.logger.info(f"Robot state saved for photo {photo_index + 1}")
            return robot_state
            
        except Exception as e:
            self.logger.error(f"Failed to save robot state for photo {photo_index + 1}: {e}")
            return None
    
    def save_hdf5_data(self, photo_index, robot_state, captured_images):
        """
        Save robot state and camera images to HDF5 file for one photo.
        
        Args:
            photo_index (int): Current photo index
            robot_state (dict): Robot state data
            captured_images (dict): Captured image data from cameras
            
        Returns:
            str: Path to saved HDF5 file
        """
        try:
            hdf5_filename = f"photo_{photo_index + 1:02d}.h5"
            hdf5_path = os.path.join(self.output_dir, hdf5_filename)
            
            self.logger.info(f"Saving HDF5 data to: {hdf5_path}")
            
            with h5py.File(hdf5_path, 'w') as f:
                # Save robot state
                robot_group = f.create_group('robot_state')
                robot_group.attrs['photo_index'] = photo_index + 1
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
                    
                    # Save color image
                    camera_group.create_dataset('color_image', 
                                              data=image_data['color_image'],
                                              compression='gzip', 
                                              compression_opts=6)
                    
                    self.logger.info(f"Saved color image for camera {serial_number} to HDF5")
                
                # Add metadata
                f.attrs['created_timestamp'] = datetime.now().isoformat()
                f.attrs['photo_index'] = photo_index + 1
                f.attrs['target_position'] = self.target_position
                f.attrs['num_cameras'] = len(captured_images)
                f.attrs['camera_serial_numbers'] = list(captured_images.keys())
            
            self.logger.info(f"HDF5 file saved successfully: {hdf5_path}")
            return hdf5_path
            
        except Exception as e:
            self.logger.error(f"Failed to save HDF5 data for photo {photo_index + 1}: {e}")
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
        Moves to target position and takes photos with user confirmation.
        """
        self.logger.info("Starting single position camera calibration workflow...")
        
        # Connect to robot and camera
        if not self.connect_robot():
            self.logger.error("Failed to connect to robot. Exiting.")
            return False
        
        if not self.connect_camera():
            self.logger.error("Failed to connect to camera. Exiting.")
            return False
        
        # Move to target position
        if not self.move_to_target_position():
            self.logger.error("Failed to move to target position. Exiting.")
            return False
        
        print(f"\n=== Fanuc Single Position Camera Calibration ===")
        print(f"Target cameras: {self.target_cameras}")
        print(f"Target position: {self.target_position}")
        print(f"Total photos to take: {self.num_photos}")
        print(f"Output directory: {self.output_dir}")
        print("Mode: USER CONFIRMATION - Press ENTER for each photo")
        print("=" * 60)
        
        # Take photos with user confirmation
        for photo_index in range(self.num_photos):
            print(f"\n--- Photo {photo_index + 1}/{self.num_photos} ---")
            
            # Wait for user confirmation
            if not wait_for_keypress():
                print("User requested to quit. Stopping calibration.")
                break
            
            # Capture images from all cameras
            captured_images = self.capture_image(photo_index)
            if captured_images is None:
                self.logger.error(f"Failed to capture images for photo {photo_index + 1}")
                continue
            
            # Save robot state
            robot_state = self.save_robot_state(photo_index)
            if robot_state is None:
                self.logger.error(f"Failed to save robot state for photo {photo_index + 1}")
                continue
            
            # Save data to HDF5 file
            hdf5_path = self.save_hdf5_data(photo_index, robot_state, captured_images)
            if hdf5_path is None:
                self.logger.error(f"Failed to save HDF5 data for photo {photo_index + 1}")
                continue
            
            # Store summary data for JSON output
            self.calibration_data["photos_taken"].append({
                "photo_index": photo_index + 1,
                "hdf5_file": hdf5_path,
                "cameras": list(captured_images.keys()),
                "timestamp": robot_state['timestamp']
            })
            self.calibration_data["robot_states"].append(robot_state)
            
            self.logger.info(f"Completed photo {photo_index + 1}")
            print(f"Photo {photo_index + 1} saved successfully!")
            print(f"HDF5 file: {hdf5_path}")
            print(f"Cameras captured: {list(captured_images.keys())}")
        
        # Save all calibration data
        calibration_file = self.save_calibration_data()
        
        print(f"\n=== Calibration Complete ===")
        print(f"Captured {len(self.calibration_data['photos_taken'])} photos")
        print(f"Target position: {self.target_position}")
        print(f"Target cameras: {self.target_cameras}")
        print(f"Data saved to: {self.output_dir}")
        print(f"Calibration file: {calibration_file}")
        
        self.logger.info("Single position camera calibration workflow completed")
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
            except Exception as e:
                self.logger.error(f"Error cleaning up cameras: {e}")
        
        self.logger.info("Cleanup completed")


def main():
    """Main function to run the calibration."""
    calibration = None
    
    try:
        # Create calibration instance
        calibration = FanucSinglePositionCalibration()
        
        # Option to set custom target position
        print("Default target position: [0, 0, 0, 0, -90, 0, 0]")
        use_custom = input("Use custom position? (y/n): ").lower().strip()
        
        if use_custom == 'y':
            print("Enter joint positions (J1, J2, J3, J4, J5, J6, gripper):")
            try:
                j1 = float(input("J1 (degrees): "))
                j2 = float(input("J2 (degrees): "))
                j3 = float(input("J3 (degrees): "))
                j4 = float(input("J4 (degrees): "))
                j5 = float(input("J5 (degrees): "))
                j6 = float(input("J6 (degrees): "))
                gripper = int(input("Gripper state (0/1): "))
                
                custom_position = [j1, j2, j3, j4, j5, j6, gripper]
                calibration.set_target_position(custom_position)
                
            except ValueError:
                print("Invalid input. Using default position.")
        
        # Run calibration
        success = calibration.run_calibration()
        
        if success:
            print("\nCalibration completed successfully!")
        else:
            print("\nCalibration failed!")
        
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user")
    
    except Exception as e:
        print(f"\nError during calibration: {e}")
    
    finally:
        # Cleanup
        if calibration:
            calibration.cleanup()


if __name__ == "__main__":
    main()






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


def wait_for_keypress():
    """
    Wait for a specific keypress to continue.
    Returns True if space is pressed, False for 'q' to quit.
    """
    while True:
        try:
            response = input("Press ENTER to take photo, 'q' + ENTER to quit: ").lower().strip()
            
            if response == '':  # Just ENTER pressed
                print("Taking photo...")
                return True
            elif response == 'q':
                print("Quitting...")
                return False
            else:
                print("Please press ENTER to take photo or 'q' + ENTER to quit")
                
        except KeyboardInterrupt:
            print("\nQuitting...")
            return False
        except Exception as e:
            print(f"Input error: {e}")
            print("Please try again...")
            continue

