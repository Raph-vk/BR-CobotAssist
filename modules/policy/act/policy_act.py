# my_app/modules/policy_interface.py

import time
import os
import sys
import glob
import numpy as np
import logging
import json
import pickle
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing.shared_memory as shared_memory
import struct
import threading
import queue
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from utils.utils import setup_logging, load_config, get_data_path
from .act_utils import load_data, compute_dict_mean, detach_dict, set_seed
from .detr.models import ACTPolicy


class PolicyInterface:
    """
    A interface that listens for start/stop recording commands,
    and policys the robot_position and master_position (and gripper flags)
    from shm_joint_data.
    """

    def __init__(self, policy_interface_commup, policy_interface_commdown, color_buffers2, depth_buffers2, shm_target_pos2_info, shm_joint_data2, shm_cpp_joint_data2_info, config, logger_pi):
        self.policy_interface_commup = policy_interface_commup
        self.policy_interface_commdown = policy_interface_commdown
        
        # Recreate actual CameraRingBuffer objects from info dictionaries
        from modules.camera.cam_utils import CameraRingBuffer
        
        self.color_buffers2 = {}
        for camera_name, buffer_info in color_buffers2.items():
            self.color_buffers2[camera_name] = CameraRingBuffer(
                name=buffer_info["name"],
                width=buffer_info["width"],
                height=buffer_info["height"],
                channels=buffer_info["channels"],
                capacity=buffer_info["capacity"],
                create=False  # Connect to existing buffer
            )
            
        self.depth_buffers2 = {}
        for camera_name, buffer_info in depth_buffers2.items():
            self.depth_buffers2[camera_name] = CameraRingBuffer(
                name=buffer_info["name"],
                width=buffer_info["width"],
                height=buffer_info["height"],
                channels=buffer_info["channels"],
                capacity=buffer_info["capacity"],
                create=False  # Connect to existing buffer
            )

        self.shm_joint_data2 = shm_joint_data2
        self.shm_cpp_joint_data2_info = shm_cpp_joint_data2_info
        self.config = config
        self.logger_pi = logger_pi

        # Attach to shm_target_pos2 shared memory segment
        self.shm_target_pos2_info = shm_target_pos2_info
        self.shm_target_pos2 = None
        if shm_target_pos2_info:
            try:
                self.shm_target_pos2 = shared_memory.SharedMemory(name=shm_target_pos2_info['name'])
                self.shm_target_pos2_capacity = shm_target_pos2_info['capacity']
                self.shm_target_pos2_entry_format = shm_target_pos2_info['entry_format']
                self.shm_target_pos2_entry_size = shm_target_pos2_info['entry_size']
                print(f"Policy: Attached to shm_target_pos2: {shm_target_pos2_info['name']}")
            except Exception as e:
                print(f"Policy: Failed to attach to shm_target_pos2: {e}")
                self.shm_target_pos2 = None

        # Attach to shm_cpp_joint_data2 shared memory segment if using C++ control loop
        self.shm_cpp_joint_data2 = None
        self.shm_cpp_joint_data2_reader = None
        if shm_cpp_joint_data2_info:
            try:
                self.shm_cpp_joint_data2 = shared_memory.SharedMemory(name=shm_cpp_joint_data2_info['name'])
                self.shm_cpp_joint_data2_capacity = shm_cpp_joint_data2_info['capacity']
                self.shm_cpp_joint_data2_slot_format = shm_cpp_joint_data2_info['slot_format']
                self.shm_cpp_joint_data2_slot_size = shm_cpp_joint_data2_info['slot_size']
                self.shm_cpp_joint_data2_header_format = shm_cpp_joint_data2_info['header_format']
                self.shm_cpp_joint_data2_header_size = shm_cpp_joint_data2_info['header_size']
                # Initialize ring buffer reader for C++ shared memory
                from utils.utils import RingBufferReader
                self.shm_cpp_joint_data2_reader = RingBufferReader(config, "shm_joint_data2")
                self.logger_pi.info(f"Policy: Attached to shm_cpp_joint_data2: {shm_cpp_joint_data2_info['name']}")
            except Exception as e:
                self.logger_pi.error(f"Policy: Failed to attach to shm_cpp_joint_data2: {e}")
                self.shm_cpp_joint_data2 = None

        self.running = False

        # Configuration-derived values (computed once for efficiency)
        self.record_divisor = self.config.get("general", {}).get("record_divisor", 4)
        self.robot_dof = self.config.get("hardware", {}).get("robot", {}).get("dof", 6)
        self.robot_dof_ee = self.config.get("hardware", {}).get("robot", {}).get("dof_ee", 1)
        self.total_dof = self.robot_dof + self.robot_dof_ee  # joints + gripper
        
        # Log configuration values for debugging
        self.logger_pi.info(f"Policy configuration: {self.robot_dof} joints + {self.robot_dof_ee} gripper = {self.total_dof} total DOF, record_divisor = {self.record_divisor}")

        # Training status tracking
        self.is_training = False
        self.current_epoch = 0
        self.training_loss = None
        self.validation_loss = None
        self.training_error = None

        # Load policy model
        self.policy_model = None
        self.policy_config = None
        self.dataset_stats = None

        self.start_position = self.config["general"]["start_position"]       

    ###################################################################
    # Commands for policy interface
    ###################################################################

    def run_policy(self, full_message):
        """
        Start the policy process.
        """
        self.logger_pi.info("Starting policy execution")
        err_msg = None       
        self.running = True

        # Load the policy in the background during initialization
        try:
            result = self.load_policy(full_message)
            if result is None:
                self.logger_pi.error("Policy loading failed - cannot start execution")
                self.running = False
                return "Policy loading failed"
        except Exception as e:
            self.logger_pi.error(f"Failed to load policy: {e}")
            self.running = False
            return f"Failed to load policy: {str(e)}"

        # Only start the policy loop if loading succeeded
        if self.running:
            self.policy_thread = threading.Thread(target=self._policy_execution_loop, daemon=True)
            self.policy_thread.start()
        
        return err_msg

    def train_policy(self, full_message):
        """
        Start training a new ACT policy model in a separate thread.
        """
        self.logger_pi.info("Starting policy training")
        
        # Check if already training
        if self.is_training:
            error_msg = "Training already in progress"
            self.logger_pi.error(error_msg)
            return error_msg
        
        # Check if policy execution is currently running
        if self.running:
            error_msg = "Cannot start training while policy execution is running"
            self.logger_pi.error(error_msg)
            return error_msg
        
        # Store the full_message for the training thread
        self.train_message = full_message
        
        # Start training in a separate thread to avoid blocking
        self.training_thread = threading.Thread(target=self._train_policy_thread, daemon=True)
        self.training_thread.start()
        
        return None  # Success - training started

    def stop(self):
        """
        Stop the policy process and clean up shared memory resources.
        """
        self.logger_pi.info("Stopping policy execution")
        self.running = False
        
        # Stop training if running
        if self.is_training:
            self.logger_pi.info("Stopping training...")
            self.is_training = False
        
        # Wait for policy thread to finish
        if hasattr(self, 'policy_thread') and self.policy_thread.is_alive():
            self.policy_thread.join(timeout=1.0)
        
        # Wait for training thread to finish  
        if hasattr(self, 'training_thread') and self.training_thread.is_alive():
            self.training_thread.join(timeout=2.0)
            
        # Clean up shared memory to prevent BufferError
        # Close camera buffers first
        if hasattr(self, 'color_buffers2'):
            for camera_name, buffer in self.color_buffers2.items():
                try:
                    buffer.close()
                    self.logger_pi.debug(f"Closed color buffer for {camera_name}")
                except Exception as e:
                    self.logger_pi.warning(f"Error closing color buffer for {camera_name}: {e}")
            self.color_buffers2.clear()
            
        if hasattr(self, 'depth_buffers2'):
            for camera_name, buffer in self.depth_buffers2.items():
                try:
                    buffer.close()
                    self.logger_pi.debug(f"Closed depth buffer for {camera_name}")
                except Exception as e:
                    self.logger_pi.warning(f"Error closing depth buffer for {camera_name}: {e}")
            self.depth_buffers2.clear()
        
        # Close other shared memory
        if self.shm_target_pos2:
            try:
                self.shm_target_pos2.close()
                self.shm_target_pos2 = None
                self.logger_pi.info("Shared memory cleaned up successfully")
            except Exception as e:
                self.logger_pi.warning(f"Error cleaning up shared memory: {e}")
                
        # Close C++ shared memory if exists
        if hasattr(self, 'shm_cpp_joint_data2') and self.shm_cpp_joint_data2:
            try:
                self.shm_cpp_joint_data2.close()
                self.shm_cpp_joint_data2 = None
                self.logger_pi.info("C++ shared memory cleaned up successfully")
            except Exception as e:
                self.logger_pi.warning(f"Error cleaning up C++ shared memory: {e}")
        
        self.logger_pi.info("Policy execution stopped")
        
        # Wait for policy thread to finish
        if hasattr(self, 'policy_thread') and self.policy_thread:
            self.policy_thread.join(timeout=2.0)
            self.logger_pi.info("Policy thread stopped")



    ###################################################################
    # Data gathering and processing run policy
    ###################################################################

    def _policy_execution_loop(self, write_chunks_to_csv=True):
        """
        Main policy execution loop that runs in a separate thread.
        """
        self.logger_pi.info("Policy execution loop started")
        frame_count = 0

        if write_chunks_to_csv:
            # Initialize execution logging
            self._init_execution_logging()
        
        while self.running:
            try:
                # 1. Receive joint information from shm_joint_data2 and gather the last 0.1 seconds of position data
                joint_data_window = self._gather_joint_data()
                
                # 2. Retrieve the latest images from the color_buffers2 and depth_buffers2
                latest_images, image_timestamps = self._retrieve_latest_images()
                
                # 3. Find the closest matching joint information based on the average timestamps of the images
                joint_position, start_seq_id = self._find_matching_joint_data(joint_data_window, image_timestamps)
                
                # 4. Run the policy on the joint information and images (simplified version)
                predicted_actions = self.execute_policy(joint_position, latest_images)
                

                # log all predicted actions
                if predicted_actions is not None:
                    self.logger_pi.debug(f"Predicted actions for frame {frame_count}: {predicted_actions}")

                # 5. interpolated actions
                interpolated_actions = self._interpolate_actions(start_seq_id, predicted_actions, joint_position)

                # 6. Update the shm_target_pos2 with predictions
                end_seq_id = self._update_target_positions(start_seq_id, interpolated_actions)

                # 7. Log execution data if wanted
                if write_chunks_to_csv:
                    self._log_execution_data(frame_count, start_seq_id, latest_images, joint_position, predicted_actions)

                # 8. Increment frame
                self.logger_pi.debug(f"Policy cycle {frame_count} completed: seq_id {start_seq_id}-{end_seq_id}")
                frame_count += 1

                # Sleep to control execution rate
                time.sleep(0.06)  # Adjust sleep time as needed for your application
                
            except Exception as e:
                self.logger_pi.error(f"Error in policy execution cycle {frame_count}: {e}")
                # print traceback of the error

                time.sleep(0.1)
                
        self.logger_pi.info(f"Policy execution loop ended after {frame_count} cycles")


    def _init_execution_logging(self):
        """Initialize execution logging files."""
        # Add execution logging path
        start_time = datetime.now().replace(microsecond=0).isoformat()
        start_time = start_time.replace(":", "-")
        self.execution_log_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "logs", f"policy_execution_{start_time}.csv")

        try:
            # Create CSV file with headers
            with open(self.execution_log_csv, 'w', newline='') as f:
                csv_writer = csv.writer(f)
                # Write CSV headers
                headers = [
                    'frame_count',
                    'timestamp',
                    'start_seq_id',
                    'latest_images',
                    'joint_position',
                    'predicted_actions'
                ]
                csv_writer.writerow(headers)
            
            self.logger_pi.info(f"Execution logging initialized:")
            self.logger_pi.info(f"  CSV log path:  {self.execution_log_csv}")
            
        except Exception as e:
            self.logger_pi.error(f"Failed to initialize execution logging: {e}")

    def _log_execution_data(self, frame_count, start_seq_id, latest_images, joint_position, predicted_actions):
        """Log execution data for this cycle."""
        try:
            timestamp = time.time()
            
            # Write to CSV file
            with open(self.execution_log_csv, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_row = [
                    frame_count,
                    timestamp,
                    start_seq_id,
                    list(latest_images),
                    list(joint_position),
                    list(predicted_actions)
                ]
                csv_writer.writerow(csv_row)
            
            # Log every 100 frames to main logger
            if frame_count % 100 == 0:
                self.logger_pi.info(f"Logged frame {frame_count}: seq_id={start_seq_id}")
                
        except Exception as e:
            self.logger_pi.error(f"Failed to log execution data for frame {frame_count}: {e}")

    def _gather_joint_data(self):
        """
        Gather joint information from shm_joint_data2 (Python queue) or shm_cpp_joint_data2 (C++ shared memory) for the last 0.1 seconds.
        """
        
        joint_data_window = []
        current_time = time.time()
        # self.logger_pi.debug("Gathering joint data for the last 0.1 seconds")

        # Use C++ shared memory if available, otherwise use Python queue
        if self.shm_cpp_joint_data2_reader is not None:
            try:
                # Read all available entries from C++ shared memory ring buffer
                all_entries = self.shm_cpp_joint_data2_reader.read_available()
                
                for entry in all_entries:
                    # Extract relevant information
                    seq_id = entry.get("seq_id", 0)
                    timestamp = entry.get("robot_position_timestamp", current_time)
                    positions = entry.get("robot_position", self.start_position)
                    
                    # Only keep data from the last 0.1 seconds
                    if timestamp > current_time - 0.1:
                        joint_data_window.append({
                            'seq_id': seq_id,
                            'timestamp': timestamp,
                            'positions': positions,
                        })
                        
                # self.logger_pi.debug(f"Gathered {len(joint_data_window)} joint data entries from C++ shared memory")
                        
            except Exception as e:
                self.logger_pi.error(f"Failed to gather joint data from C++ shared memory: {e}")
                return []
                
        elif self.shm_joint_data2 is not None:
            # Read from Python queue (existing implementation)
            try:    
                # Read all available joint data from the queue without blocking
                while True:
                    try:
                        # Get joint data from queue (non-blocking)
                        joint_data = self.shm_joint_data2.get_nowait()
                        
                        self.logger_pi.debug(f"Received joint data: {joint_data}")

                        # Extract relevant information
                        seq_id = joint_data.get("seq_id", 0)
                        timestamp = joint_data.get("robot_position_timestamp", current_time)
                        positions = joint_data.get("robot_position", self.start_position)
                        
                        # Only keep data from the last 0.1 seconds
                        if timestamp > current_time - 0.1:
                            joint_data_window.append({
                                'seq_id': seq_id,
                                'timestamp': timestamp,
                                'positions': positions,
                            })
                        
                    except queue.Empty:
                        # No more data in queue
                        break
                        
            except Exception as e:
                self.logger_pi.error(f"Failed to gather joint data from Python queue: {e}")
                return []
        else:
            self.logger_pi.warning("No joint data source available (neither C++ shared memory nor Python queue)")
            return []
        
        # self.logger_pi.debug(f"Gathered {len(joint_data_window)} joint data entries from the last 0.1 seconds")
        return joint_data_window

    def _retrieve_latest_images(self):
        """
        Retrieve the latest images from color_buffers2 and depth_buffers2.
        """
        latest_images = {}
        image_timestamps = {}
        
        # self.logger_pi.debug("Retrieving latest images from buffers")

        # Get images from color buffers
        if hasattr(self, 'color_buffers2') and self.color_buffers2:
            # self.logger_pi.debug("Retrieving color images from buffers")
            for camera_name, color_buffer in self.color_buffers2.items():
                try:
                    color_data = color_buffer.read()
                    if color_data:
                        color_image = color_data["image"]
                        color_timestamp = color_data["timestamp"]
                        latest_images[f"{camera_name}_color"] = color_image
                        image_timestamps[f"{camera_name}_color"] = color_timestamp
                        # self.logger_pi.debug(f"Retrieved color image from {camera_name} at {color_timestamp}")
                except Exception as e:
                    self.logger_pi.warning(f"Failed to read color image from {camera_name}: {e}")
        
        # Get images from depth buffers
        if hasattr(self, 'depth_buffers2') and self.depth_buffers2:
            for camera_name, depth_buffer in self.depth_buffers2.items():
                try:
                    depth_data = depth_buffer.read()
                    if depth_data:
                        depth_image = depth_data["image"]
                        depth_timestamp = depth_data["timestamp"]
                        latest_images[f"{camera_name}_depth"] = depth_image
                        image_timestamps[f"{camera_name}_depth"] = depth_timestamp
                except Exception as e:
                    self.logger_pi.warning(f"Failed to read depth image from {camera_name}: {e}")
        
        return latest_images, image_timestamps

    def _find_matching_joint_data(self, joint_data_window, image_timestamps):
        """
        Find the closest matching joint information based on the average timestamps of the images.
        Returns a sequence ID (integer) for the starting position.
        """
        if not image_timestamps:
            # If no images, use current timestamp to generate a seq_id
            self.logger_pi.warning("No image timestamps available, using 0 seq_id")
            return self.start_position, 0
        
        avg_image_timestamp = np.mean(list(image_timestamps.values()))
        
        # Find the joint data entry with timestamp closest to average image timestamp
        if joint_data_window:
            closest_joint = min(joint_data_window, 
                               key=lambda x: abs(x['timestamp'] - avg_image_timestamp))
            closest_joint_position = closest_joint['positions']
            closest_seq_id = closest_joint['seq_id']
            return closest_joint_position, closest_seq_id
        else:
            # If no joint data, generate a seq_id from timestamp
            self.logger_pi.warning("No joint data available, using 0 seq_id")
            return self.start_position, 0

    def _update_target_positions(self, start_seq_id, predicted_actions):
        """
        Update the shm_target_pos2 with predictions.
        """
        if not self.shm_target_pos2 or predicted_actions is None or len(predicted_actions) == 0:
            return start_seq_id
        
        try:
            # Write predicted actions to shared memory
            for action_data in predicted_actions:
                seq_id = action_data['seq_id']
                action = action_data['action']
                
                # Validate action dimensions before packing
                if len(action) != self.total_dof:  # Expected: joints + gripper
                    self.logger_pi.warning(f"Action has {len(action)} elements, expected {self.total_dof} ({self.robot_dof} joints + {self.robot_dof_ee} gripper)")
                
                # Pack data according to format: sequence_id (I) + joint positions (d each)
                data = struct.pack(self.shm_target_pos2_entry_format, seq_id, *action)
                
                # Calculate buffer position (circular buffer)
                buffer_index = seq_id % self.shm_target_pos2_capacity
                offset = buffer_index * self.shm_target_pos2_entry_size
                
                # Write to shared memory
                self.shm_target_pos2.buf[offset:offset+len(data)] = data
            
            end_seq_id = predicted_actions[-1]['seq_id']
            # self.logger_pi.debug(f"Updated shm_target_pos2 with actions {start_seq_id} to {end_seq_id}")
            return end_seq_id
            
        except Exception as e:
            self.logger_pi.error(f"Failed to update shm_target_pos2: {e}")
            return start_seq_id

    def _interpolate_actions(self, start_seq_id, predicted_actions, current_joint_position):
        """
        Interpolate actions to provide smooth trajectories at full control frequency.
        
        The policy predicts actions every record_divisor control cycles (e.g., every 4 cycles at 62.5 Hz),
        but we need to provide smooth control commands at the full control frequency (250 Hz).
        
        Args:
            start_seq_id: Starting sequence ID
            predicted_actions: List of predicted actions from policy (at reduced frequency)
            current_joint_position: Current joint position to start interpolation from
            
        Returns:
            List of interpolated actions at full control frequency
        """
        if predicted_actions is None or len(predicted_actions) == 0:
            return []
            
        interpolated_actions = []
        
        # Start with current position as the first reference point
        prev_action = np.array(current_joint_position)
        
        # Interpolate between consecutive predicted actions
        for i in range(len(predicted_actions)):
            next_action = predicted_actions[i]  # Access numpy array directly
            
            # Ensure next_action has correct dimensions (total_dof elements: joints + gripper)
            if len(next_action) != self.total_dof:
                self.logger_pi.warning(f"Predicted action has {len(next_action)} elements, expected {self.total_dof}")
                if len(next_action) < self.total_dof:
                    next_action = np.pad(next_action, (0, self.total_dof - len(next_action)), 'constant')
                else:
                    next_action = next_action[:self.total_dof]
            
            # Generate interpolated steps between prev_action and next_action
            for step in range(1, self.record_divisor):
                # Linear interpolation factor (0.0 to 1.0)
                alpha = step / self.record_divisor
                
                # Interpolate between previous and next action
                interpolated_action = (1 - alpha) * prev_action + alpha * next_action
                
                # Create action dictionary with interpolated values
                interpolated_seq_id = start_seq_id + i * self.record_divisor + step
                interpolated_actions.append({
                    'seq_id': interpolated_seq_id,
                    'action': interpolated_action.tolist()
                })
            
            # Update prev_action for next iteration
            prev_action = next_action
        
        # self.logger_pi.debug(f"Interpolated {len(predicted_actions)} predicted actions into {len(interpolated_actions)} smooth actions (DOF: {self.robot_dof}+{self.robot_dof_ee}={self.total_dof})")
        return interpolated_actions

    ###################################################################
    # Helper functions
    ###################################################################

    def _find_latest_model(self, dataset_dir):
        """
        Find the most recent model in the dataset Models directory.
        Returns the model name (directory name) or None if no models found.
        """
        models_dir = os.path.join(dataset_dir, "Models")
        if not os.path.isdir(models_dir):
            return None
        
        model_dirs = [d for d in os.listdir(models_dir) 
                     if os.path.isdir(os.path.join(models_dir, d))]
        if not model_dirs:
            return None
        
        # Sort by directory name (which should be timestamp-based)
        model_dirs.sort(reverse=True)  # Most recent first
        return model_dirs[0]

    def _get_num_episodes(self, dataset_dir):
        """Count the number of .hdf5 files in the dataset directory."""
        num_episodes = 0
        if os.path.exists(dataset_dir):
            for filename in os.listdir(dataset_dir):
                if filename.endswith(".hdf5"):
                    num_episodes += 1
        return num_episodes
    
    def _get_camera_names_from_dataset(self, dataset_dir):
        """Get camera names from the first available HDF5 file in the dataset directory."""
        camera_names = []
        
        if not os.path.exists(dataset_dir):
            return camera_names
            
        for filename in os.listdir(dataset_dir):
            if filename.endswith(".hdf5"):
                full_path = os.path.join(dataset_dir, filename)
                try:
                    with h5py.File(full_path, 'r') as file:
                        # Check if the images group exists
                        if 'images' in file:
                            # Get camera names from the images group
                            camera_names = list(file['images'].keys())
                            self.logger_pi.info(f"Found cameras in {filename}: {camera_names}")
                            break  # Use first valid file
                        else:
                            self.logger_pi.warning(f"No 'images' group found in {filename}")
                except Exception as e:
                    self.logger_pi.warning(f"Could not read camera names from {filename}: {e}")
                    continue
                    
        return camera_names
    
    def _save_config(self, ckpt_dir, model_config):
        """Save model-specific training configuration to config.json."""
        config_path = os.path.join(ckpt_dir, 'config.json')
        
        # Create a copy of the config with relative paths for saving
        config_to_save = deepcopy(model_config)
        
        # Convert absolute paths to relative paths for portability
        app_directory = self.config["general"]["app_directory"]
        data_directory = self.config["general"]["data_directory"]
        
        # Make ckpt_dir relative to app_directory
        if config_to_save['ckpt_dir'].startswith(app_directory):
            rel_ckpt_dir = os.path.relpath(config_to_save['ckpt_dir'], app_directory)
            config_to_save['ckpt_dir'] = rel_ckpt_dir
            config_to_save['policy_config']['ckpt_dir'] = rel_ckpt_dir
        
        current_config_str = json.dumps(config_to_save, sort_keys=True, indent=4)
        
        if os.path.exists(config_path):
            # If config exists, verify it matches
            with open(config_path, 'r') as file:
                existing_config_str = file.read()
            existing_config = json.loads(existing_config_str)

            discrepancies = []
            for key, new_value in config_to_save.items():
                old_value = existing_config.get(key)
                if new_value != old_value:
                    discrepancies.append(f"{key}: old value = {old_value}, new value = {new_value}")

            if discrepancies:
                self.logger_pi.error("Model configuration mismatch:")
                for discrepancy in discrepancies:
                    self.logger_pi.error(discrepancy)
                return False
        else:
            # Save new config
            with open(config_path, 'w') as file:
                file.write(current_config_str)
            self.logger_pi.info(f"Saved model config to {config_path}")
        
        return True
    
    def _load_training_data(self, dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
        """Load training data using act_utils."""
        self.logger_pi.info(f"Loading training data from {dataset_dir}")
        self.logger_pi.info(f"Episodes: {num_episodes}, Cameras: {camera_names}")
        
        try:
            train_dataloader, val_dataloader, stats = load_data(
                dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, self.config, self.logger_pi
            )
            
            self.logger_pi.info("Training data loaded successfully")
            self.logger_pi.info(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")
            
            return train_dataloader, val_dataloader, stats
            
        except Exception as e:
            self.logger_pi.error(f"Error loading training data: {e}")
            raise
    

    ###################################################################
    # Train policy
    ###################################################################

    def _train_policy_thread(self):
        """
        Training thread function - runs the actual training process.
        """
        try:
            # Set training status
            self.is_training = True
            self.training_error = None
            
            # Get general system config from config.yaml
            app_directory = self.config["general"]["app_directory"]
            data_directory = self.config["general"]["data_directory"]
            
            # Get dataset selection from UI message
            selected_dataset = self.train_message["dataset_name"]
            
            # Get camera info from system config
            camera_info = self.config.get("hardware", {}).get("camera", {}).get("info", [])
            system_camera_names = [cam["name"] for cam in camera_info] if camera_info else []
            
            # Training parameters - these will be saved to model-specific config.json
            model_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Model-specific training parameters (will be saved to config.json)
            num_epochs = 15000
            batch_size = 16
            chunk_size = 75
            seed = 0
            continue_training = True
            
            # Policy architecture parameters (will be saved to config.json)
            lr = 1e-5
            kl_weight = 100
            hidden_dim = 1024
            dim_feedforward = 1024
            state_dim = 7
            enc_layers = 6
            dec_layers = 8
            nheads = 32
            
            # Backbone parameters (will be saved to config.json)
            backbone = "resnet34"
            lr_backbone = 1e-5
            
            # Setup directories using selected dataset
            dataset_dir = get_data_path(self.config, selected_dataset)
            ckpt_dir = os.path.join(dataset_dir, "Models", model_name)
            
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
                
            self.logger_pi.info(f"Training on dataset: {selected_dataset}")
            self.logger_pi.info(f"Model directory: {ckpt_dir}")
                
            # Get dataset info from actual data files
            num_episodes = self._get_num_episodes(dataset_dir)
            if num_episodes == 0:
                self.logger_pi.error(f'No episodes found in {dataset_dir}')
                self.training_error = "No training data found"
                return
                
            if num_episodes < batch_size:
                self.logger_pi.error(f'Not enough episodes ({num_episodes}) for batch size {batch_size}')
                self.training_error = f"Need at least {batch_size} episodes, found {num_episodes}"
                return
        
            # Use camera names from dataset, fallback to system config if needed
            camera_names = self._get_camera_names_from_dataset(dataset_dir)
            if not camera_names:
                camera_names = system_camera_names
                self.logger_pi.warning(f"No camera names found in dataset, using system config: {camera_names}")
            else:
                self.logger_pi.info(f"Camera names from dataset: {camera_names}")
            
            # Create model-specific policy config (this goes into config.json)
            policy_config = {
                'lr': lr,
                'num_queries': chunk_size,
                'kl_weight': kl_weight,
                'hidden_dim': hidden_dim,
                'dim_feedforward': dim_feedforward,
                'lr_backbone': lr_backbone,
                'backbone': backbone,
                'enc_layers': enc_layers,
                'dec_layers': dec_layers,
                'nheads': nheads,
                'camera_names': camera_names,
                'ckpt_dir': ckpt_dir,  # This will be relative path in the saved config
                'chunk_size': chunk_size,
            }
            
            # Create full model-specific config (this goes into config.json)
            model_config = {
                'num_epochs': num_epochs,
                'num_episodes': num_episodes,
                'ckpt_dir': ckpt_dir,  # This will be made relative when saved
                'state_dim': state_dim,
                'lr': lr,
                'policy_config': policy_config,
                'seed': seed,
                'camera_names': camera_names,
                'continue_training': continue_training,
            }
            
            # Load data - this will need act_utils functionality
            self.logger_pi.info("Loading training data...")
            train_dataloader, val_dataloader, stats = self._load_training_data(
                dataset_dir, num_episodes, camera_names, batch_size, batch_size)
            
            # Save dataset stats
            stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
            with open(stats_path, 'wb') as f:
                pickle.dump(stats, f)
                
            # Save model-specific config.json
            if not self._save_config(ckpt_dir, model_config):
                self.training_error = "Config validation failed"
                return
                
            # Start training
            self.logger_pi.info("Starting training process...")
            best_ckpt_info = self._train_bc(train_dataloader, val_dataloader, model_config)
            
            if best_ckpt_info:
                best_epoch, min_val_loss, _ = best_ckpt_info
                self.logger_pi.info(f"Training completed. Best model at epoch {best_epoch} with val loss {min_val_loss:.6f}")
                self.training_error = None  # Success
            else:
                self.training_error = "Training failed or was interrupted"
                
        except Exception as e:
            self.logger_pi.error(f"Error in policy training: {e}")
            import traceback
            self.logger_pi.error(traceback.format_exc())
            self.training_error = f"Training error: {str(e)}"
        
        finally:
            # Always reset training status when done
            self.is_training = False

    def _train_bc(self, train_dataloader, val_dataloader, config):
        """Main training loop - full implementation matching the original train_bc."""
        continue_training = config['continue_training']
        ckpt_dir = config['ckpt_dir']
        seed = config['seed']
        policy_config = config['policy_config']
        max_epochs = config['num_epochs']
        loaded_epoch = 0

        self.logger_pi.info(f"Starting training: {max_epochs} epochs, continue_training={continue_training}")
        
        # Set training status
        self.is_training = True
        self.current_epoch = 0
        
        set_seed(seed)
        policy = self.make_policy(policy_config)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            policy.cuda()
            self.logger_pi.info("Policy moved to GPU")
        else:
            self.logger_pi.warning("GPU not available, using CPU")

        # Continue training from existing checkpoint if requested
        if continue_training:
            best_epoch = 0
            if os.path.isdir(ckpt_dir):
                for filename in os.listdir(ckpt_dir):
                    if filename.startswith('policy_best_epoch') and filename.endswith('.ckpt'):
                        try:
                            epoch = int(filename.split('_')[-1].split('.')[0])
                            if epoch > best_epoch:
                                best_epoch = epoch
                                loaded_epoch = best_epoch + 1
                        except ValueError:
                            continue
                            
                if best_epoch > 0:
                    self.logger_pi.info(f'Found existing checkpoint: best_epoch {best_epoch}')
                    joined_path = os.path.join(ckpt_dir, f'policy_best_epoch_{best_epoch}.ckpt')
                    policy.load_state_dict(torch.load(joined_path, map_location='cpu'))
                    self.logger_pi.info(f'Loaded checkpoint from epoch {best_epoch} to continue training')

        optimizer = self.make_optimizer(policy)

        train_history = []
        validation_history = []
        min_val_loss = np.inf
        best_ckpt_info = None
        previous_best_epoch = loaded_epoch - 1 if continue_training else 0
        start_epoch = loaded_epoch if continue_training else 0

        try:
            for epoch in tqdm(range(start_epoch, max_epochs), desc="Training Progress"):
                self.current_epoch = epoch
                
                # Check if training should be stopped
                if not self.is_training:
                    self.logger_pi.info("Training stopped by user request")
                    break
                
                self.logger_pi.info(f'Epoch {epoch}')
                
                # Validation phase
                with torch.no_grad():
                    policy.eval()
                    epoch_dicts = []
                    
                    for batch_idx, data in enumerate(val_dataloader):
                        forward_dict = self._forward_pass(data, policy)
                        forward_dict = detach_dict(forward_dict)
                        epoch_dicts.append(forward_dict)
                        
                    epoch_summary = compute_dict_mean(epoch_dicts)
                    validation_history.append(epoch_summary)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    epoch_val_loss = epoch_summary['loss']
                    self.validation_loss = epoch_val_loss
                    
                    # Save best checkpoint
                    if epoch_val_loss < min_val_loss:
                        min_val_loss = epoch_val_loss
                        best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
                        best_epoch = epoch
                        best_ckpt_path = os.path.join(ckpt_dir, f'policy_best_epoch_{best_epoch}.ckpt')
                        torch.save(policy.state_dict(), best_ckpt_path)
                        self.logger_pi.info(f'Saved best ckpt to {best_ckpt_path}, val loss {min_val_loss:.6f} @ epoch{best_epoch}')
                        
                        # Clean up previous best checkpoint
                        if previous_best_epoch >= 0 and previous_best_epoch != best_epoch:
                            previous_best_ckpt_path = os.path.join(ckpt_dir, f'policy_best_epoch_{previous_best_epoch}.ckpt')
                            if os.path.exists(previous_best_ckpt_path):
                                os.remove(previous_best_ckpt_path)
                                self.logger_pi.info(f'Deleted previous best ckpt {previous_best_ckpt_path}')
                        previous_best_epoch = best_epoch

                self.logger_pi.info(f'Val loss:   {epoch_val_loss:.5f}')
                summary_string = ' '.join([f'{k}: {v.item():.3f}' for k, v in epoch_summary.items()])
                self.logger_pi.info(summary_string)

                # Training phase
                policy.train()
                optimizer.zero_grad()
                
                for batch_idx, data in enumerate(train_dataloader):
                    forward_dict = self._forward_pass(data, policy)
                    loss = forward_dict['loss']
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    forward_dict = detach_dict(forward_dict)
                    forward_dict = {k: v.cpu() for k, v in forward_dict.items()}
                    train_history.append(forward_dict)

                # Calculate training epoch summary
                epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*(epoch-start_epoch):(batch_idx+1)*((epoch-start_epoch)+1)])
                epoch_train_loss = epoch_summary['loss']
                self.training_loss = epoch_train_loss
                
                self.logger_pi.info(f'Train loss: {epoch_train_loss:.5f}')
                summary_string = ' '.join([f'{k}: {v.item():.3f}' for k, v in epoch_summary.items()])
                self.logger_pi.info(summary_string)
                
                # Plot training curves
                self._plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

                # Save periodic checkpoints
                if epoch % 500 == 0:
                    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
                    torch.save(policy.state_dict(), ckpt_path)
                    self.logger_pi.info(f'Saved periodic checkpoint: {ckpt_path}')

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            self.logger_pi.error(f"Error during training: {e}")
            raise
        finally:
            self.is_training = False

        if best_ckpt_info:
            best_epoch, min_val_loss, best_state_dict = best_ckpt_info
            self.logger_pi.info(f"Training completed. Best epoch: {best_epoch}, Best val loss: {min_val_loss:.6f}")
        
        return best_ckpt_info

    def execute_policy(self, joint_position, latest_images):
        """
        Execute the actual ACT policy for inference.
        Returns a list of predicted actions (without sequence IDs - they will be assigned during interpolation).
        
        Args:
            start_seq_id: Starting sequence ID (used for logging, not assigned to actions here)
            joint_position: Current joint positions
            latest_images: Latest camera images
            
        Returns:
            List of predicted actions (just the action values, no seq_id)
        """
        # Check if policy is loaded
        if self.policy_config is None:
            self.logger_pi.error("Policy config not loaded - cannot execute policy")
            return []
        
        if self.policy_model is None:
            self.logger_pi.error("Policy model not loaded - cannot execute policy")
            return []
        
        # Prepare images - combine color and depth for each camera
        image_list = []
        camera_names = self.policy_config['camera_names']
        
        for cam_name in camera_names:
            color_key = f"{cam_name}_color"
            depth_key = f"{cam_name}_depth"
            
            # Get color image (3 channels: RGB)
            if color_key in latest_images:
                color_image = latest_images[color_key]
                if isinstance(color_image, np.ndarray):
                    color_image = torch.from_numpy(color_image).float().permute(2, 0, 1) / 255.0
            else:
                color_image = torch.zeros(3, 240, 424)  # RGB dummy
                self.logger_pi.warning(f"No color image data for {cam_name}, using dummy")
            
            # Get depth image (1 channel: Depth)
            if depth_key in latest_images:
                depth_image = latest_images[depth_key]
                if isinstance(depth_image, np.ndarray):
                    # Convert depth to single channel and normalize
                    if len(depth_image.shape) == 3:
                        depth_image = depth_image[:, :, 0]  # Take first channel if multi-channel
                    depth_image = torch.from_numpy(depth_image).float().unsqueeze(0) / 65535.0  # Normalize depth by max 16-bit value
            else:
                depth_image = torch.zeros(1, 240, 424)  # Depth dummy
                self.logger_pi.warning(f"No depth image data for {cam_name}, using dummy")
            
            # Combine RGB + D = 4 channels
            rgbd_image = torch.cat([color_image, depth_image], dim=0)  # Shape: [4, 240, 424]
            image_list.append(rgbd_image)
        
        # Stack images into batch
        if image_list:
            images = torch.stack(image_list).unsqueeze(0)  # Add batch dimension -> [1, num_cameras, 4, 240, 424]
        else:
            # Fallback dummy images with 4 channels (RGBD)
            num_cameras = len(camera_names)
            images = torch.zeros(1, num_cameras, 4, 240, 424)
        
        # Prepare joint positions - preprocessing like the working code
        qpos = torch.tensor(joint_position, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        # Normalize joint positions using dataset stats
        stats = self.dataset_stats
        joint_pos_mean = np.array(stats['joint_pos_mean'])
        joint_pos_std = np.array(stats['joint_pos_std'])
        
        # Convert to torch tensors and normalize (following the working code pattern)
        mean = torch.tensor(joint_pos_mean, dtype=torch.float32)
        std = torch.tensor(joint_pos_std, dtype=torch.float32)
        qpos = (qpos - mean) / std
        
        # Move to device
        device = next(self.policy_model.parameters()).device
        images = images.to(device)
        qpos = qpos.to(device)
        
        # Run inference
        with torch.no_grad():
            raw_actions = self.policy_model(qpos, images)  # Get predicted actions
            actions = raw_actions
            
        # Convert back to numpy and denormalize
        actions = actions.cpu().numpy()[0]  # Remove batch dimension
        
        # Denormalize actions using the same method as the working code
        # Check for both possible key formats in dataset_stats
        if 'action_mean' in self.dataset_stats and 'action_std' in self.dataset_stats:
            action_mean = np.array(self.dataset_stats['action_mean'])
            action_std = np.array(self.dataset_stats['action_std'])
            actions = actions * action_std + action_mean
            # self.logger_pi.debug(f"Denormalized actions using action_mean/action_std: mean={action_mean}, std={action_std}")
        elif 'action_pos_mean' in self.dataset_stats and 'action_pos_std' in self.dataset_stats:
            action_mean = np.array(self.dataset_stats['action_pos_mean'])
            action_std = np.array(self.dataset_stats['action_pos_std'])
            actions = actions * action_std + action_mean
            # self.logger_pi.debug(f"Denormalized actions using action_pos_mean/action_pos_std: mean={action_mean}, std={action_std}")
        else:
            self.logger_pi.error(f"No action normalization stats found. Available keys: {list(self.dataset_stats.keys())}")
            self.logger_pi.error(f"Raw actions (non-denormalized): {actions}")

        return actions  # Return numpy array directly - more efficient


    ###################################################################
    # Policy and policy helpers
    ###################################################################

    def load_policy(self, full_message):
        """
        Load the ACT policy from model-specific config.json and checkpoint.
        Uses general config.yaml for system paths only.
        """
        self.logger_pi.info(f"load_policy called with full_message: {full_message}")
        
        try:
            # Get system paths from config.yaml
            app_directory = self.config["general"]["app_directory"]
            data_directory = self.config["general"]["data_directory"]
            
            # Get selected dataset and model from message if provided
            if full_message is None:
                full_message = {}
            selected_dataset = full_message["dataset_name"]
            selected_model = full_message.get("model_name", None)  # Will auto-find if None
            
            # If no model specified, find the most recent one
            dataset_path = get_data_path(self.config, selected_dataset)
            if selected_model is None:
                selected_model = self._find_latest_model(dataset_path)
                if selected_model is None:
                    self.logger_pi.error(f"No models found in dataset: {selected_dataset}")
                    return None
                self.logger_pi.info(f"Auto-selected most recent model: {selected_model}")
            
            model_directory = os.path.join(dataset_path, "Models", selected_model)
            
            if not os.path.isdir(model_directory):
                self.logger_pi.error(f"No model directory found: {model_directory}")
                return None
                
            self.logger_pi.info(f"Found model directory: {model_directory}")
            self.logger_pi.info(f"Loading dataset: {selected_dataset}, model: {selected_model}")
            
            # Load model-specific config.json
            config_path = os.path.join(model_directory, 'config.json')
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            
            # Load dataset stats
            stats_path = os.path.join(model_directory, 'dataset_stats.pkl')
            with open(stats_path, 'rb') as f:
                self.dataset_stats = pickle.load(f)
            
            # Get policy config from the loaded model config
            policy_config = model_config.get('policy_config', {})
            if not policy_config:
                self.logger_pi.error("No policy_config found in config.json")
                return None
            
            # Use model_directory as checkpoint directory
            ckpt_dir = model_directory
            
            # Find the best epoch checkpoint
            best_epoch = 0
            for filename in os.listdir(ckpt_dir):
                if filename.startswith('policy_best_epoch') and filename.endswith('.ckpt'):
                    try:
                        epoch_str = filename.replace('policy_best_epoch_', '').replace('.ckpt', '')
                        epoch = int(epoch_str)
                        best_epoch = max(best_epoch, epoch)
                    except ValueError:
                        continue

            if best_epoch <= 0:
                self.logger_pi.error("No valid checkpoint found.")
                return None

            ckpt_name = f"policy_best_epoch_{best_epoch}.ckpt"
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            
            if not os.path.exists(ckpt_path):
                self.logger_pi.error(f"Checkpoint file does not exist: {ckpt_path}")
                return None

            self.logger_pi.info(f'Loading checkpoint: {ckpt_path}')

            # Create policy model
            policy = self.make_policy(policy_config)
            
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            loading_status = policy.load_state_dict(checkpoint)
            self.logger_pi.info(f"Checkpoint loading status: {loading_status}")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                policy.cuda()
                self.logger_pi.info("Policy moved to GPU")
            
            policy.eval()
            self.logger_pi.info(f'Policy loaded successfully from: {ckpt_path}')
            
            self.policy_model = policy
            self.policy_config = policy_config
            return policy
            
        except Exception as e:
            self.logger_pi.error(f"Failed to load policy: {e}")
            import traceback
            self.logger_pi.error(traceback.format_exc())
            return None

    def make_policy(self, policy_config):
        """
        Initializes the policy model.
        """
        policy = ACTPolicy(policy_config)
        return policy

    def make_optimizer(self, policy):
        """
        Configures the optimizer for the policy.
        """
        optimizer = policy.configure_optimizers()
        return optimizer

    def _plot_history(self, train_history, validation_history, epoch, ckpt_dir, seed):
        """Plot training curves - adapted from the original training script."""
        # Save training curves
        if not train_history or not validation_history:
            return
            
        for key in train_history[0]:
            plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
            plt.figure()
            
            train_values = [summary[key].item() if hasattr(summary[key], 'item') else summary[key] 
                          for summary in train_history]
            val_values = [summary[key].item() if hasattr(summary[key], 'item') else summary[key] 
                        for summary in validation_history]
            
            plt.plot(np.linspace(0, epoch-1, len(train_history)), train_values, label='train')
            plt.plot(np.linspace(0, epoch-1, len(validation_history)), val_values, label='validation')
            plt.ylim([0, 2])
            plt.tight_layout()
            plt.legend()
            plt.title(key)
            plt.savefig(plot_path)
            plt.close()
            
        self.logger_pi.info(f"Saved training plots to {ckpt_dir}")

    def _forward_pass(self, data, policy):
        """Forward pass for training - matches the original forward_pass function."""
        image_data, joint_pos_data, action_data, is_pad = data
        
        # Move data to GPU if available
        if torch.cuda.is_available():
            image_data = image_data.cuda()
            joint_pos_data = joint_pos_data.cuda() 
            action_data = action_data.cuda()
            is_pad = is_pad.cuda()
            
        return policy(joint_pos_data, image_data, action_data, is_pad)






def send_response(logger_si, policy_interface_commup, payload, error="None", **kwargs):
    """
    Build a response dict and push it on policy_interface_commup.

      {
        "type"   : "RESP",
        "message": <e.g. "start_teleoperation_record" | "stop" …>,
        "error"  : <error string or "None">,
        ...      : any extra fields supplied via **kwargs
      }
    """
    response = payload.copy()
    response["type"] = "RESP"

    # Normalise / inject error
    if error not in ("None", ""):
        response["error"] = error
    elif response.get("error", "") == "":
        response["error"] = "None"

    # Add any extras
    response.update(kwargs)

    # Log & publish
    logger_si.info(f"Preparing to send response: {response}")
    policy_interface_commup.put(response)
    logger_si.info(f"Sent response: {response}")



def run_policy_interface(policy_interface_commup, policy_interface_commdown, color_buffers2, depth_buffers2, shm_target_pos2_info, shm_joint_data2, shm_cpp_joint_data2_info=None):
    """
    Spawn a policyInterface instance and service controller commands.
    """
    component_tag = "POLICY_INTERFACE"
    logger_pi = setup_logging(component_tag)
    logger_pi.info("Starting policy Interface…")

    # Load configuration
    config = load_config()

    # Instantiate interface
    try:
        policy = PolicyInterface(policy_interface_commup, policy_interface_commdown, color_buffers2, depth_buffers2, shm_target_pos2_info, shm_joint_data2, shm_cpp_joint_data2_info, config, logger_pi)
        queue_check_period = config["general"]["check_queue_period"]
        send_response(logger_pi, policy_interface_commup,
                    {"interface": component_tag, "message": "initialization"}, error="None")     
           
    except Exception as e:
        err = f"Failed to initialise policyInterface: {e}"
        logger_pi.error(err)
        send_response(logger_pi, policy_interface_commup,
                      {"interface": component_tag, "message": "initialization"}, error=err)
        return


    while True:
        if not policy_interface_commdown.empty():
            full_message = policy_interface_commdown.get()
            logger_pi.info("Received message: %s", full_message)

            msg_type      = full_message.get("type", "")
            msg_interface = full_message.get("interface", "")
            message       = full_message.get("message", "")

            if msg_type == "CMD" and msg_interface == component_tag:
                if message == "stop":
                    try:
                        policy.stop()
                        send_response(logger_pi, policy_interface_commup,
                                      full_message, error="None")
                    except Exception as e:
                        send_response(logger_pi, policy_interface_commup,
                                      full_message, error=str(e))
                    break  # exit process loop


                elif message == "run_policy":
                    if policy.running or policy.is_training:
                        send_response(logger_pi, policy_interface_commup,
                                      full_message, error="Already recording")
                        continue
                    err = policy.run_policy(full_message)
                    send_response(logger_pi, policy_interface_commup,
                                  full_message, error=("None" if err is None else err))
                    

                elif message == "train_policy":
                    if policy.running or policy.is_training:
                        send_response(logger_pi, policy_interface_commup,
                                      full_message, error="Already busy with executing or training policy")
                        continue
                    err = policy.train_policy(full_message)
                    send_response(logger_pi, policy_interface_commup,
                                  full_message, error=("None" if err is None else err))

                else:
                    send_response(logger_pi, policy_interface_commup,
                                  full_message, error="Unknown command")

            else:
                send_response(logger_pi, policy_interface_commup,
                              full_message, error="Unknown message")

        time.sleep(queue_check_period)





