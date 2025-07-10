# my_app/modules/save_interface.py

import time
import json
import threading
from threading import Thread
import os
import sys
import struct
import numpy as np
import base64
import h5py
from datetime import datetime
from multiprocessing import shared_memory
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.utils import setup_logging, load_config, RingBufferReader, get_data_path


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def _next_ep_idx(dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    existing = [f for f in os.listdir(dataset_dir) if f.startswith("episode_") and f.endswith(".hdf5")]
    nums = [int(f.split('_')[1].split('.')[0]) for f in existing if f.split('_')[1].split('.')[0].isdigit()]
    return max(nums, default=-1) + 1

class SaveInterface:
    """
    A interface that listens for start/stop recording commands,
    and saves the robot_position and teachbot_position (and gripper flags)
    from shm_joint_data.
    """

    def __init__(self, save_interface_commup, save_interface_commdown, shm_joint_data, logger_si, config, color_buffers1=None, depth_buffers1=None):
        """
        :param save_interface_commup: multiprocessing.Queue, from interface to controller
        :param save_interface_commdown: multiprocessing.Queue, from controller to interface
        :param shm_joint_data: multiprocessing.Queue, where the robot_interface (or controlling code)
                               puts dict items like:
                               {
                                 "robot_position": [...],
                                 "teachbot_position": [...],
                                 "sent_robot_position": [...],
                                "robot_position_timestamp": <float>,
                               }
        :param color_buffers1: dict of CameraRingBuffer, color images from cameras
        :param depth_buffers1: dict of CameraRingBuffer, depth images from cameras
        """
        self.save_interface_commup = save_interface_commup
        self.save_interface_commdown = save_interface_commdown
        self.shm_joint_data = shm_joint_data
        self.logger_si = logger_si
        self.config = config
        self.color_buffers1 = color_buffers1 or {}
        self.depth_buffers1 = depth_buffers1 or {}

        # Flags & State
        self.recording = False
        self.record_thread = None
        
        self.shm_reader = None

        # We'll store everything in a dict so that the final JSON is more structured:
        self.recorded_data = {}

        # Basic timing/loop rates
        self.control_dt = self.config["hardware"]["robot"]["control_dt"]
        self.status_refresh_period = self.config["general"]["status_refresh_period"]
        self.queue_check_period = self.config["general"]["check_queue_period"]
        self.check_queue_period_divisor = self.config["general"]["check_queue_period_divisor"]
        self.default_recording_speed = self.config["general"]["default_recording_speed"]
        self.dof = self.config["hardware"]["robot"]["dof"] + self.config["hardware"]["robot"]["dof_ee"]  # Total DOF including gripper
        self.control_loop_language = self.config["general"]["control_loop_language"]
        self.record_duration = self.config["general"]["record_duration"]
        self.data_directory = self.config["general"]["data_directory"]
        self.record_divisor = self.config["general"].get("record_divisor", 4)  # Default to 4 if not specified


        # Output file for final results
        self.output_filename = None

        self.logger_si.info("SaveInterface created.")

    ###################################################################
    # Save interface commands
    ###################################################################

    def start_teleoperation_record(self, full_message):
        """
        Begin reading from shm_joint_data in a background thread.
        """
        self.logger_si.info("start_teleoperation_record invoked...")
        if self.recording:
            self.logger_si.warning("start_teleoperation_record invoked, but already recording.")
            return

        # Create the output directory if it doesn't exist
        filename = full_message["recording_name"]
        recording_speed = full_message.get("recording_speed", self.default_recording_speed)
        filename = get_data_path(self.config, filename)

        # if it exists, error
        if os.path.exists(filename):
            self.logger_si.error("File already exists: %s", filename)
            return
        else:
            self.output_filename = filename

        # Clear old data (fresh each time)
        self.recorded_data = {
            "metadata": {
                "format_version": 1,
                "start_time": time.time(),
                "end_time": None,
                "recording_speed": recording_speed,
                "dof": self.config["hardware"]["robot"]["dof"],
                "dof_ee": self.config["hardware"]["robot"]["dof_ee"],
                "total_dof": self.dof,
                "control_dt": self.control_dt,
                "control_loop_language": self.control_loop_language
            },
            "samples": []
        }

        if self.control_loop_language == "cpp":
            self.logger_si.info("Initializing RingBufferReader for C++ shared memory (start_teleoperation_record)")
            try:
                self.shm_reader = RingBufferReader(self.config, "shm_joint_data1")
                self.logger_si.info("RingBufferReader initialized successfully")
                self.logger_si.info("SLOT_FMT: %s, SLOT_SIZE: %d bytes", 
                                  self.shm_reader.SLOT_FMT, self.shm_reader.SLOT_SIZE)
            except Exception as e:
                self.logger_si.error("Failed to initialize RingBufferReader: %s", str(e))
                import traceback
                self.logger_si.error("Traceback: %s", traceback.format_exc())
                self.shm_reader = None            

        self.recording = True
        self.record_thread = threading.Thread(target=self.record_loop, daemon=True)
        self.record_thread.start()

        self.logger_si.info("start_teleoperation_record successful")

    def record_episodes(self, full_message):
        """
        Begin reading from shm_joint_data in a background thread.
        This is a placeholder for future functionality.
        """
        self.logger_si.info("record_episodes invoked...")
        if self.recording:
            self.logger_si.warning("record_episodes invoked, but already recording.")
            return

        # Create the output directory if it doesn't exist
        folder_name = full_message["dataset_name"]
        recording_speed = full_message.get("recording_speed", self.default_recording_speed)
        folder_name = get_data_path(self.config, folder_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name, exist_ok=True)
            self.logger_si.info("Created directory: %s", folder_name)

        # Set the output directory for episode saving
        self.output_filename = folder_name  # Store the directory path directly
        self.episode_idx = _next_ep_idx(folder_name)

        # Clear old data (fresh each time)
        self.recorded_data = {
            "metadata": {
                "format_version": 1,
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": None,
                "recording_speed": recording_speed,
                "dof": self.config["hardware"]["robot"]["dof"],
                "dof_ee": self.config["hardware"]["robot"]["dof_ee"],
                "total_dof": self.dof,
                "control_dt": self.control_dt,
                "control_loop_language": self.control_loop_language
            },
            "samples": []
        }


        self.recording = True
        self.recording_speed = full_message.get("recording_speed", self.default_recording_speed)

        if self.control_loop_language == "cpp":
            self.logger_si.info("Initializing RingBufferReader for C++ shared memory (record_episodes)")
            try:
                self.shm_reader = RingBufferReader(self.config, "shm_joint_data1")
                self.logger_si.info("RingBufferReader initialized successfully")
                self.logger_si.info("SLOT_FMT: %s, SLOT_SIZE: %d bytes", 
                                  self.shm_reader.SLOT_FMT, self.shm_reader.SLOT_SIZE)
            except Exception as e:
                self.logger_si.error("Failed to initialize RingBufferReader: %s", str(e))
                import traceback
                self.logger_si.error("Traceback: %s", traceback.format_exc())
                self.shm_reader = None     

        self.record_thread = threading.Thread(target=self.record_loop_img, daemon=True)
        self.record_thread.start()

        self.logger_si.info("record_episodes successful, recording started.")

    def record_episode(self, full_message):
        """
        Begin reading from shm_joint_data in a background thread.
        This is a placeholder for future functionality.
        """
        self.record_one_episode = True
        self.record_episodes(full_message)

    def stop(self):
        """
        Cleanly disconnect/stop this interface:
          1) Drain any leftover items from the shm_joint_data queue.
          2) If we are still recording, stop and save final data.
        """
        self.logger_si.info("Disconnect invoked...")

        # 2) If still recording, stop now (this saves final data)
        self.logger_si.info("Recording status: %s", self.recording)
        if self.recording:
            self.stop_recording()

        self.logger_si.info("Disconnect completed.")


    ####################################################################
    # Save interface helper functions
    ####################################################################

    def record_loop(self):
        """
        Record loop that only captures robot joint data, no images.
        """
        while self.recording:
            try:
                # Try to get a piece of joint data from the queue
                if self.control_loop_language == "cpp":
                    # Use the C++ shared memory reader
                    data_item = next(self.shm_reader)
                else:
                    data_item = self.shm_joint_data.get(timeout=0.004)

                if data_item is None:
                    self.logger_si.warning("Received None from shm_joint_data, skipping.")
                    continue

                self.logger_si.info("Received data item: %s", data_item)
                
                # Extract all fields from the data structure to match the control loop format
                robot_pos = data_item.get("robot_position", [])
                teachbot_pos = data_item.get("teachbot_position", [])
                sent_robot_position = data_item.get("sent_robot_position", [])
                robot_position_timestamp = data_item.get("robot_position_timestamp", time.time())
                seq_id = data_item.get("seq_id", 0)

                # Build a sample record (no images) with all available fields
                # Note: gripper state is now included as last element in position arrays
                sample = {
                    "robot_position": robot_pos,
                    "teachbot_position": teachbot_pos,
                    "sent_robot_position": sent_robot_position,
                    "robot_position_timestamp": robot_position_timestamp,
                    "seq_id": seq_id
                }

                self.recorded_data["samples"].append(sample)
                
            except Exception:
                # Possibly queue.Empty or other minor errors
                pass

    def record_loop_img(self):
        """
        Record loop that captures both robot joint data and camera images.
        """

        self.logger_si.info("Starting image recording loop...")

        # empty buffers if they are not empty
        if self.control_loop_language == "cpp":
            while True:
                data_item = next(self.shm_reader, None)
                if data_item == None:
                    break
        else:
            while not self.shm_joint_data.empty():
                try:
                    data_item = self.shm_joint_data.get(timeout=0.1)
                except Exception:
                    break


        while self.recording:
            try:
                episode_name = f"episode_{self.episode_idx}.hdf5"
                self.logger_si.info("Recording episode: %s", episode_name)

                self.episode_timesteps = int(self.record_duration / self.control_dt)
                self.episode_joint_buffer = []

                # Wait for first joint state data
                first_data_item = None
                while self.recording and first_data_item is None:
                    try:
                        if self.control_loop_language == "cpp":
                            if self.shm_reader is None:
                                self.logger_si.error("shm_reader is None in C++ mode!")
                                break
                            first_data_item = next(self.shm_reader)
                        else:
                            first_data_item = self.shm_joint_data.get(timeout=2)
                    except Exception as e:
                        self.logger_si.info("Waiting for first joint data item, retrying...")
                        pass
                if not self.recording:
                    break
                    
                # Process the first data item
                if first_data_item:
                    required_fields = ["robot_position_timestamp", "robot_position", "teachbot_position", "sent_robot_position"]
                    if all(field in first_data_item for field in required_fields):
                        joint_data = {field: first_data_item[field] for field in required_fields}
                        self.episode_joint_buffer.append(joint_data)

                # set metadata start_time for episode
                self.recorded_data["metadata"]["start_time"] = time.time()                    

                # Adjust the loop range if we already processed the first item
                remaining_timesteps = self.episode_timesteps - len(self.episode_joint_buffer)
                for t in tqdm(range(remaining_timesteps), desc="Collecting joint data"):
                
                    if not self.recording:
                        self.logger_si.info("Recording stopped, exiting loop.")
                        break

                    # get the last item from the queue
                    if self.control_loop_language == "cpp":
                        # Use the C++ shared memory reader
                        data_item = next(self.shm_reader)
                    else:
                        data_item = self.shm_joint_data.get(timeout=1)

                    # collect the data item and its information
                    joint_data = {
                        "robot_position_timestamp": data_item["robot_position_timestamp"],
                        "robot_position": data_item["robot_position"],
                        "teachbot_position": data_item["teachbot_position"],
                        "sent_robot_position": data_item["sent_robot_position"],
                    }
    
                    self.episode_joint_buffer.append(joint_data)

                self.logger_si.info("Collected joint data for episode %s", self.episode_idx)

                # Collect all images from ring buffers into buffers
                color_image_buffers = {cam_name: [] for cam_name in self.color_buffers1.keys()}
                depth_image_buffers = {cam_name: [] for cam_name in self.depth_buffers1.keys()}
                
                self._collect_images_from_buffers(color_image_buffers, depth_image_buffers, self.episode_joint_buffer)
                
                # Get camera names
                camera_names = list(set(list(self.color_buffers1.keys()) + list(self.depth_buffers1.keys())))
                
                # Align and organize data with record_divisor consideration
                data_dict = self._align_and_organize_episode_data(
                    self.episode_joint_buffer, 
                    color_image_buffers, 
                    depth_image_buffers, 
                    camera_names, 
                    self.record_divisor
                ) 

                # Save the data to an HDF5 file
                # the metadata of the file should contain timestamp, the joint state number to which the images correspond, the serial number of each camera and cam name
                self._save_episode_to_hdf5(data_dict, camera_names)

                if self.record_one_episode:
                    self.logger_si.info("Recording one episode, stopping after this.")
                    # send stop command to the robot_controller
                    message = {"type": "CMD", "interface": "SAVE_INTERFACE", "message": "stop"}
                    self.save_interface_commup.put(message)
                    break

            except Exception as e:
                self.logger_si.error("Error during recording loop: %s", str(e))

        self.logger_si.info("Image recording loop ended.")

    def stop_recording(self, full_message=None):
        """
        Stop reading from the shared memory queue and save the recorded data.
        """
        if not self.recording:
            self.logger_si.warning("Stop_recording invoked, but not currently recording.")
            if full_message is not None:
                error_msg = "Not currently recording"
                return error_msg

        # Stop the record loop
        self.recording = False
        if self.record_thread is not None:
            self.record_thread.join(timeout=2.0)
            self.record_thread = None

        # Mark the end_time in metadata
        self.recorded_data["metadata"]["end_time"] = time.time()

        # Example: save to a JSON file
        try:
            # check if there is recorded data
            if not self.recorded_data["samples"]:
                self.logger_si.warning("No data recorded, nothing to save.")
                return

            os.makedirs(os.path.dirname(self.output_filename), exist_ok=True)
            self.logger_si.info("Saving data to %s", self.output_filename)
            with open(self.output_filename, "w") as f:
                json.dump(self.recorded_data, f, indent=2, cls=NumpyEncoder)
            self.logger_si.info("Data saved successfully to %s", self.output_filename) 

        except Exception as e:
            self.logger_si.error("[ERROR][SAVE_INTERFACE]: Failed to save data: %s", e)

        if self.control_loop_language == "cpp":
            # Clean up the C++ shared memory segment
            try:
                shm = shared_memory.SharedMemory(name=self.shm_reader.shm.name, create=False)
                shm.close()
                shm.unlink()  # Remove the shared memory segment
                self.logger_si.info("C++ shared memory segment cleaned up.")
            except Exception as e:
                self.logger_si.error("Failed to clean up C++ shared memory: %s", e)
        return

    ######################################################################
    # Helper functions for image collection and alignment
    ######################################################################

    def _find_closest_image(self, img_list, target_ts):
        """
        Find the image with timestamp closest to target timestamp.
        """
        if not img_list:
            self.logger_si.warning(f"[find_closest_image] No images available for target timestamp: {target_ts}")
            return None
        
        # Find the image with the minimum timestamp difference
        closest = min(img_list, key=lambda x: abs(x['timestamp'] - target_ts))
        
        # If timestamp difference is more than 0.05 seconds, print a warning
        if abs(closest['timestamp'] - target_ts) > 0.05:
            self.logger_si.warning(f"[find_closest_image] Timestamp difference is more than 0.05 seconds: {abs(closest['timestamp'] - target_ts)}")
            
        
        return closest

    def _collect_images_from_buffers(self, color_image_buffers, depth_image_buffers, joint_buffer):
        """
        Collect images from the camera ring buffers into buffers, but only those within the relevant timestamp range.
        Only retrieves images from the last timestamp before the first joint data item 
        up to and including the first image after the last joint data item.
        """
        if not joint_buffer:
            self.logger_si.warning("No joint data available, skipping image collection")
            return
            
        # Get timestamp range from joint buffer
        joint_timestamps = [item['robot_position_timestamp'] for item in joint_buffer]
        first_joint_ts = min(joint_timestamps)
        last_joint_ts = max(joint_timestamps)
        
        # Add small buffer to capture images slightly outside the range
        ts_buffer = 0.1  # 100ms buffer
        min_ts_threshold = first_joint_ts - ts_buffer
        max_ts_threshold = last_joint_ts
                
        # Collect color images within timestamp range
        for camera_name, color_buffer in self.color_buffers1.items():
            # Use optimized ring buffer method
            try:
                images_in_range = color_buffer.read_all_within_range(min_ts_threshold, max_ts_threshold)
                for image_data in images_in_range:
                    image_entry = {
                        'timestamp': image_data['timestamp'],
                        'image': image_data['image']
                    }
                    color_image_buffers[camera_name].append(image_entry)
            except Exception as e:
                self.logger_si.warning(f"Could not collect color images from {camera_name}: {e}")

        # Collect depth images within timestamp range  
        for camera_name, depth_buffer in self.depth_buffers1.items():
            # Use optimized ring buffer method
            try:
                images_in_range = depth_buffer.read_all_within_range(min_ts_threshold, max_ts_threshold)
                for image_data in images_in_range:
                    image_entry = {
                        'timestamp': image_data['timestamp'],
                        'image': image_data['image']
                    }
                    depth_image_buffers[camera_name].append(image_entry)
            except Exception as e:
                self.logger_si.warning(f"Could not collect depth images from {camera_name}: {e}")
                    
        # Report on min and max timestamps of collected images
        for cam_name in color_image_buffers:
            # Report color image timestamps
            if not color_image_buffers[cam_name]:
                self.logger_si.warning(f"No color images collected for {cam_name}")
            # Report depth image timestamps
            if not depth_image_buffers[cam_name]:
                self.logger_si.warning(f"No depth images collected for {cam_name}")
                    
    def _align_and_organize_episode_data(self, joint_buffer, color_image_buffers, depth_image_buffers, camera_names, record_divisor):
        """
        Align and organize joint data with images, using record_divisor to determine which timesteps get images.
        """
        # Sort all data by timestamp
        joint_buffer.sort(key=lambda x: x['robot_position_timestamp'])
        for cam_name in camera_names:
            if cam_name in color_image_buffers:
                color_image_buffers[cam_name].sort(key=lambda x: x['timestamp'])
            if cam_name in depth_image_buffers:
                depth_image_buffers[cam_name].sort(key=lambda x: x['timestamp'])

        # create a dictionary with sub levels joint_data and images, each joint_data will have a timestamp, robot_position, teachbot_position and sent_robot_position
        data_dict = {
            'joint_data': [],
            'images': {}
        }
        
        # Initialize image subdictionaries for each camera
        for cam_name in camera_names:
            data_dict['images'][cam_name] = {
                'color': [],
                'depth': [],
                'color_timestamps': [],
                'depth_timestamps': []
            }

        # Iterate through joint data and align images
        for i, joint_data in enumerate(joint_buffer):
            try:
                # Extract joint data fields
                robot_position_timestamp = joint_data['robot_position_timestamp']
                robot_position = joint_data['robot_position']
                teachbot_position = joint_data['teachbot_position']
                sent_robot_position = joint_data['sent_robot_position']

                # Create a joint data entry
                joint_entry = {
                    'robot_position_timestamp': robot_position_timestamp,
                    'robot_position': robot_position,
                    'teachbot_position': teachbot_position,
                    'sent_robot_position': sent_robot_position
                }
                
                # Add to the joint data list
                data_dict['joint_data'].append(joint_entry)

                # Only process every record_divisor-th joint data item
                if i % record_divisor != 0:
                    continue
                
                # Align images for this timestamp
                for cam_name in camera_names:
                    color_images = color_image_buffers.get(cam_name, [])
                    depth_images = depth_image_buffers.get(cam_name, [])
                    
                    # Find closest color image
                    closest_color_image = self._find_closest_image(color_images, robot_position_timestamp)
                    if closest_color_image is not None:
                        data_dict['images'][cam_name]['color'].append(closest_color_image['image'])
                        data_dict['images'][cam_name]['color_timestamps'].append(closest_color_image['timestamp'])
                    
                    # Find closest depth image
                    closest_depth_image = self._find_closest_image(depth_images, robot_position_timestamp)
                    if closest_depth_image is not None:
                        data_dict['images'][cam_name]['depth'].append(closest_depth_image['image'])
                        data_dict['images'][cam_name]['depth_timestamps'].append(closest_depth_image['timestamp'])

            except Exception as e:
                self.logger_si.error(f"Error processing joint data item {i}: {e}")
                import traceback
                self.logger_si.error(f"Traceback: {traceback.format_exc()}")
                continue  # Skip this item and continue with the next one
        return data_dict

    def _save_episode_to_hdf5(self, data_dict, camera_names):
        """
        Save the aligned episode data to an HDF5 file with metadata.
        """
        try:
            # Create the episode filename based on episode index
            episode_filename = f"episode_{self.episode_idx}.hdf5"
            
            # Handle both directory path and file path in output_filename
            if self.output_filename and os.path.isdir(self.output_filename):
                # output_filename is a directory (for episode recording)
                episode_filepath = os.path.join(self.output_filename, episode_filename)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(episode_filepath), exist_ok=True)
            
            with h5py.File(episode_filepath, 'w') as f:
                # Handle joint data separately - save each field as a separate dataset
                if 'joint_data' in data_dict and data_dict['joint_data']:
                    try:
                        joint_data_list = data_dict['joint_data']
                        
                        # Extract each field into separate arrays
                        robot_position_timestamps = [item['robot_position_timestamp'] for item in joint_data_list]
                        robot_positions = [item['robot_position'] for item in joint_data_list]
                        teachbot_positions = [item['teachbot_position'] for item in joint_data_list]
                        sent_robot_positions = [item['sent_robot_position'] for item in joint_data_list]

                        # Save each field as a separate dataset
                        try:
                            if not robot_position_timestamps:
                                raise ValueError("robot_position_timestamps is empty")
                            timestamps_array = np.array(robot_position_timestamps)
                            f.create_dataset('robot_position_timestamps', data=timestamps_array)
                            if not robot_positions:
                                raise ValueError("robot_positions is empty")
                            positions_array = np.array(robot_positions, dtype=np.float32)
                            f.create_dataset('robot_positions', data=positions_array)
                            if not teachbot_positions:
                                raise ValueError("teachbot_positions is empty")
                            teachbot_array = np.array(teachbot_positions, dtype=np.float32)
                            f.create_dataset('teachbot_positions', data=teachbot_array)
                            if not sent_robot_positions:
                                raise ValueError("sent_robot_positions is empty")
                            sent_array = np.array(sent_robot_positions, dtype=np.float32)
                            f.create_dataset('sent_robot_positions', data=sent_array)
                        except Exception as e:
                            self.logger_si.error(f"Failed to save: {e}. Data sample: {sent_robot_positions[0] if sent_robot_positions else 'None/Empty'}")
                    except Exception as e:
                        self.logger_si.error(f"Failed to save joint_data: {e}")
                        import traceback
                        self.logger_si.error(f"Joint data save traceback: {traceback.format_exc()}")
                
                # Handle image data separately for each camera
                if 'images' in data_dict:
                    images_group = f.create_group('images')
                    for cam_name, cam_data in data_dict['images'].items():
                        cam_group = images_group.create_group(cam_name)
                        
                        # Save color images
                        if 'color' in cam_data:
                            try:
                                color_array = np.array(cam_data['color'])
                                cam_group.create_dataset('color', data=color_array)
                            except Exception as e:
                                self.logger_si.error(f"Failed to save color images for {cam_name}: {e}. Data: {len(cam_data['color'])} items, sample type: {type(cam_data['color'][0]) if cam_data['color'] else 'None/Empty'}")
                        
                        # Save depth images
                        if 'depth' in cam_data:
                            try:
                                depth_array = np.array(cam_data['depth'])
                                cam_group.create_dataset('depth', data=depth_array)
                            except Exception as e:
                                self.logger_si.error(f"Failed to save depth images for {cam_name}: {e}. Data: {len(cam_data['depth'])} items, sample type: {type(cam_data['depth'][0]) if cam_data['depth'] else 'None/Empty'}")
                        
                        # Save color timestamps
                        if 'color_timestamps' in cam_data:
                            try:
                                color_ts_array = np.array(cam_data['color_timestamps'])
                                cam_group.create_dataset('color_timestamps', data=color_ts_array)
                            except Exception as e:
                                self.logger_si.error(f"Failed to save color timestamps for {cam_name}: {e}. Data: {len(cam_data['color_timestamps'])} items, sample: {cam_data['color_timestamps'][0] if cam_data['color_timestamps'] else 'None/Empty'}")
                        
                        # Save depth timestamps
                        if 'depth_timestamps' in cam_data:
                            try:
                                depth_ts_array = np.array(cam_data['depth_timestamps'])
                                cam_group.create_dataset('depth_timestamps', data=depth_ts_array)
                            except Exception as e:
                                self.logger_si.error(f"Failed to save depth timestamps for {cam_name}: {e}. Data: {len(cam_data['depth_timestamps'])} items, sample: {cam_data['depth_timestamps'][0] if cam_data['depth_timestamps'] else 'None/Empty'}")
                
                # Handle any other top-level data that might be simple arrays
                for key, value_list in data_dict.items():
                    if key not in ['joint_data', 'images'] and value_list:  # Skip already handled data
                        try:
                            # Convert to numpy array
                            data_array = np.array(value_list)
                            # Only save if it's not an object array
                            if data_array.dtype != 'O':
                                f.create_dataset(key, data=data_array)
                            else:
                                self.logger_si.warning(f"Skipping dataset {key} due to object dtype. Data sample: {value_list[0] if value_list else 'None/Empty'}")
                        except Exception as e:
                            self.logger_si.error(f"Failed to save dataset {key}: {e}. Data sample: {value_list[0] if value_list else 'None/Empty'}")
                
                # Add metadata
                try:
                    metadata_group = f.create_group('metadata')
                
                    # Episode metadata
                    try:
                        # Log each attribute before saving
                        episode_idx = self.episode_idx
                        start_time = self.recorded_data["metadata"]["start_time"]
                        end_time = time.time()
                        record_divisor = self.record_divisor
                        total_timesteps = len(data_dict.get('joint_data', []))
                        recording_speed = self.recorded_data["metadata"].get("recording_speed", self.default_recording_speed)
                        dof = self.recorded_data["metadata"].get("dof", self.config["hardware"]["robot"]["dof"])
                        dof_ee = self.recorded_data["metadata"].get("dof_ee", self.config["hardware"]["robot"]["dof_ee"])
                        total_dof = self.recorded_data["metadata"].get("total_dof", self.dof)
                        control_dt = self.control_dt
                        control_loop_language = self.control_loop_language
                        
                        # Handle None values with appropriate fallbacks
                        if start_time is None:
                            start_time = end_time - self.record_duration  # Estimate start time
                            self.logger_si.warning("start_time was None, using estimated value")
                        
                        # Convert to appropriate types
                        metadata_group.attrs['episode_idx'] = int(episode_idx)
                        metadata_group.attrs['start_time'] = float(start_time)
                        metadata_group.attrs['end_time'] = float(end_time)
                        metadata_group.attrs['record_divisor'] = int(record_divisor)
                        metadata_group.attrs['total_timesteps'] = int(total_timesteps)
                        metadata_group.attrs['recording_speed'] = float(recording_speed)
                        metadata_group.attrs['dof'] = int(dof)
                        metadata_group.attrs['dof_ee'] = int(dof_ee)
                        metadata_group.attrs['total_dof'] = int(total_dof)
                        metadata_group.attrs['control_dt'] = float(control_dt)
                        metadata_group.attrs['control_loop_language'] = control_loop_language
                    except Exception as e:
                        # Enhanced error logging to identify None/missing variables
                        self.logger_si.error(f"Failed to save episode metadata: {e}")
                        self.logger_si.error(f"episode_idx: {getattr(self, 'episode_idx', 'MISSING')}")
                        self.logger_si.error(f"recorded_data: {getattr(self, 'recorded_data', 'MISSING')}")
                        if hasattr(self, 'recorded_data') and self.recorded_data:
                            self.logger_si.error(f"start_time: {self.recorded_data.get('metadata', {}).get('start_time', 'MISSING')}")
                        self.logger_si.error(f"record_divisor: {getattr(self, 'record_divisor', 'MISSING')}")
                        self.logger_si.error(f"data_dict: {type(data_dict)} with keys: {list(data_dict.keys()) if isinstance(data_dict, dict) else 'NOT_DICT'}")
                        import traceback
                        self.logger_si.error(f"Traceback: {traceback.format_exc()}")
                        self.logger_si.error(f"Exception type: {type(e)}")
                        import traceback
                        self.logger_si.error(f"Traceback: {traceback.format_exc()}")
                    
                    # Camera metadata
                    try:
                        camera_metadata = metadata_group.create_group('cameras')
                        for cam_name in camera_names:
                            try:
                                cam_group = camera_metadata.create_group(cam_name)
                                cam_group.attrs['name'] = cam_name
                                
                                # Try to get camera serial from config or use camera name
                                try:
                                    cam_serial = self.config.get('cameras', {}).get(cam_name, {}).get('serial', cam_name)
                                    cam_group.attrs['serial'] = cam_serial
                                except:
                                    cam_group.attrs['serial'] = cam_name
                                
                                # Add image dimensions metadata if available from the new structure
                                if 'images' in data_dict and cam_name in data_dict['images']:
                                    cam_data = data_dict['images'][cam_name]
                                    
                                    if 'color' in cam_data and cam_data['color']:
                                        try:
                                            color_array = np.array(cam_data['color'])
                                            # Convert shape tuple to list for HDF5 compatibility
                                            cam_group.attrs['color_image_shape'] = list(color_array.shape)
                                            cam_group.attrs['color_image_count'] = len([x for x in cam_data['color'] if x is not None])
                                        except Exception as e:
                                            self.logger_si.error(f"Failed to save color metadata for {cam_name}: {e}. Data: {len(cam_data['color']) if cam_data['color'] else 'None/Empty'} items")
                                    
                                    if 'depth' in cam_data and cam_data['depth']:
                                        try:
                                            depth_array = np.array(cam_data['depth'])
                                            # Convert shape tuple to list for HDF5 compatibility
                                            cam_group.attrs['depth_image_shape'] = list(depth_array.shape)
                                            cam_group.attrs['depth_image_count'] = len([x for x in cam_data['depth'] if x is not None])
                                        except Exception as e:
                                            data_info = f"{len(cam_data['depth'])} items" if cam_data['depth'] else "None/Empty"
                                            self.logger_si.error(f"Failed to save depth metadata for {cam_name}: {e}. Data: {data_info}")
                                            
                            except Exception as e:
                                self.logger_si.error(f"Failed to save camera metadata for {cam_name}: {e}")
                    except Exception as e:
                        self.logger_si.error(f"Failed to save camera metadata: {e}")
                    
                    # Joint state metadata
                    try:
                        joint_metadata = metadata_group.create_group('joint_states')
                        if 'joint_data' in data_dict and data_dict['joint_data']:
                            joint_count = len(data_dict['joint_data'])
                            joint_metadata.attrs['joint_states_count'] = joint_count
                            # Add info about the separate joint data fields
                            # Note: gripper state is now included as last element in position arrays
                            joint_metadata.attrs['fields'] = ['robot_position_timestamps', 'robot_positions', 'teachbot_positions', 'sent_robot_positions']
                    except Exception as e:
                        self.logger_si.error(f"Failed to save joint state metadata: {e}. joint_data available: {'YES' if 'joint_data' in data_dict else 'NO'}")
                        
                except Exception as e:
                    self.logger_si.error(f"Failed to save metadata: {e}")
                    import traceback
                    self.logger_si.error(f"Metadata save traceback: {traceback.format_exc()}")
            
            self.logger_si.info(f"Successfully saved episode {self.episode_idx} to {episode_filepath}")
            
            # Reset buffers for next episode
            self.episode_joint_buffer.clear()
            
            # Increment episode index for next recording
            self.episode_idx += 1
            
        except Exception as e:
            self.logger_si.error(f"Failed to save episode {self.episode_idx} to HDF5: {e}")
            raise


def send_response(logger_si, save_interface_commup, payload, error="None", **kwargs):
    """
    Build a response dict and push it on save_interface_commup.

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
    save_interface_commup.put(response)
    logger_si.info(f"Sent response: {response}")


def run_save_interface(save_interface_commup, save_interface_commdown, shm_joint_data, color_buffers1, depth_buffers1):
    """
    Spawn a SaveInterface instance and service controller commands.
    """
    component_tag = "SAVE_INTERFACE"
    logger_si = setup_logging(component_tag)
    logger_si.info("Starting Save Interface…")

    # Load configuration
    config = load_config()

    # Instantiate interface
    try:
        saver = SaveInterface(save_interface_commup, save_interface_commdown,
                              shm_joint_data, logger_si, config, color_buffers1, depth_buffers1)
        queue_check_period = config["general"]["check_queue_period"]
        # Notify controller that we’re alive and healthy
        send_response(logger_si, save_interface_commup,
                    {"interface": component_tag, "message": "initialization"}, error="None")     
           
    except Exception as e:
        err = f"Failed to initialise SaveInterface: {e}"
        logger_si.error(err)
        send_response(logger_si, save_interface_commup,
                      {"interface": component_tag, "message": "initialization"}, error=err)
        return


    while True:
        if not save_interface_commdown.empty():
            full_message = save_interface_commdown.get()
            logger_si.info("Received message: %s", full_message)

            msg_type      = full_message.get("type", "")
            msg_interface = full_message.get("interface", "")
            message       = full_message.get("message", "")

            if msg_type == "CMD" and msg_interface == component_tag:
                if message == "stop":
                    try:
                        saver.stop()
                        send_response(logger_si, save_interface_commup,
                                      full_message, error="None")
                    except Exception as e:
                        send_response(logger_si, save_interface_commup,
                                      full_message, error=str(e))
                    break  # exit process loop

                elif message == "record_episodes":
                    if saver.recording:
                        send_response(logger_si, save_interface_commup,
                                      full_message, error="Already recording")
                        continue
                    saver.record_episodes(full_message)
                    send_response(logger_si, save_interface_commup,
                                  full_message, error="None")
                    
                elif message == "record_episode":
                    if saver.recording:
                        send_response(logger_si, save_interface_commup,
                                      full_message, error="Already recording")
                        continue
                    err = saver.record_episode(full_message)
                    send_response(logger_si, save_interface_commup,
                                  full_message, error=("None" if err is None else err))

                elif message == "start_teleoperation_record":
                    if saver.recording:
                        send_response(logger_si, save_interface_commup,
                                      full_message, error="Already recording")
                        continue
                    err = saver.start_teleoperation_record(full_message)
                    send_response(logger_si, save_interface_commup,
                                  full_message, error=("None" if err is None else err))

                else:
                    send_response(logger_si, save_interface_commup,
                                  full_message, error="Unknown command")

            else:
                send_response(logger_si, save_interface_commup,
                              full_message, error="Unknown message")

        time.sleep(queue_check_period)
