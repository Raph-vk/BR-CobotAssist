#!/usr/bin/env python3
# Teachbot controller implementation

import sys
import os
import time
import json
import importlib
import threading
import multiprocessing
import struct
from contextlib import contextmanager
from multiprocessing import shared_memory

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError
from utils.utils import RingBufferReader, get_data_path
from modules.camera.cam_utils import CameraRingBufferManager


################################################################################
# RobotController
################################################################################
class RobotController:
    def __init__(self, config, logger_tc, setup_id=None):
        """
        Initialize the RobotController with the configuration and logger.
        Args:
            config: Configuration dictionary
            logger_tc: Logger instance
            setup_id: Setup ID (e.g., "1", "2") for multi-setup routing
        """
        self.logger_tc = logger_tc
        self.config = config
        self.setup_id = setup_id
        self.success = False

        error_msg = "None"
        self.logger_tc.info(f"Initializing RobotController for setup {setup_id}...")

        # 1) Parse config (with setup-specific configuration)
        self._parse_config()
        
        # 2) Import the correct hardware interface modules dynamically
        self._load_dynamic_modules()

        # 3) Initialize local interface states
        self._init_interface_states()

        # 4) Setup RabbitMQ infrastructure (exchange, queue, binding)
        self.setup_rabbitmq_infrastructure()

        # 5) Start the command consumer (commdown)
        self.start_command_consumer()

        self.logger_tc.info("RobotController initialized successfully.")
        self.send_response({"message": "initialization"}, error=error_msg)


    ###################################################################
    # Commands / Behaviors
    ###################################################################

    def start_teleoperation(self, payload):
        """
        Connect the necessary interfaces and start teleoperation (no recording).
        """
        error_msg = "None"

        try:
            if self.has_teachbot:
                self.connect_teachbot_interface()
            self.connect_robot_interface()

            # Copy for robot interface
            robot_cmd = dict(payload)
            robot_cmd["type"] = "CMD"
            robot_cmd["interface"] = "ROBOT_INTERFACE"
            self.send_command(robot_cmd)

            self.logger_tc.info("teleoperation_running True")

        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"start_teleoperation error: {error_msg}")

        # Send response
        self.send_response(
            payload=payload,
            error=error_msg,
        )

    def get_interface_name(self, base_interface_name):
        """Get setup-specific interface name."""
        if self.setup_id:
            return f"{int(self.setup_id):02d}_{base_interface_name}"
        else:
            return base_interface_name

    def get_base_interface_name(self, interface_name):
        """Extract base interface name from setup-specific interface name."""
        if self.setup_id and interface_name.startswith(f"{int(self.setup_id):02d}_"):
            # Find the first underscore and return everything after it
            underscore_pos = interface_name.find('_')
            if underscore_pos != -1:
                return interface_name[underscore_pos + 1:]
        return interface_name

    def start_teleoperation_record(self, payload):
        """
        Connect the necessary interfaces and start teleoperation+recording.
        """
        error_msg = "None"

        try:
            self.connect_save_interface()
            if self.has_teachbot:
                self.connect_teachbot_interface()
            self.connect_robot_interface()

            # Save interface copy
            save_cmd = dict(payload)
            save_cmd["type"] = "CMD"
            save_cmd["interface"] = self.get_interface_name("SAVE_INTERFACE")
            self.send_command(save_cmd)

            # Robot interface copy
            robot_cmd = dict(payload)
            robot_cmd["type"] = "CMD"
            robot_cmd["interface"] = self.get_interface_name("ROBOT_INTERFACE")
            self.send_command(robot_cmd)

            self.logger_tc.info("teleoperation_running True")

        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"start_teleoperation_record error: {error_msg}")

        # Send response
        self.send_response(
            payload=payload,
            error=error_msg,
        )

    def play_recording(self, payload):
        """
        Connect the robot interface and play a recorded sequence.
        """
        error_msg = "None"

        try:
            self.connect_robot_interface()

            # Add type and interface
            robot_interface_cmd = dict(payload)
            robot_interface_cmd["type"] = "CMD"
            robot_interface_cmd["interface"] = "ROBOT_INTERFACE"
            self.send_command(robot_interface_cmd)

        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"play_recording error: {error_msg}")

        # Send response
        self.send_response(
            payload=payload,
            error=error_msg,
        )

    def report_recording_names(self, payload):
        """
        Gather the list of recorded files and send them as a response,
        including the recording speed if present in the file's metadata.
        """
        error_msg = "None"
        file_info_list = []

        try:
            path = get_data_path(self.config)
            file_names = os.listdir(path)

            for file_name in file_names:
                full_path = os.path.join(path, file_name)

                # Only attempt to parse JSON files
                if file_name.endswith(".json"):
                    try:
                        with open(full_path, 'r') as f:
                            data = json.load(f)
                            # retrieve recording_speed if it exists
                            rec_speed = data.get("metadata", {}).get("recording_speed", None)
                    except (json.JSONDecodeError, OSError) as e:
                        self.logger_tc.error(f"Could not load or parse {file_name}: {e}")
                        rec_speed = None

                    file_info_list.append({
                        "file_name": file_name,
                        "recording_speed": rec_speed
                    })

            self.logger_tc.info(f"Reported recording names: {file_info_list}")

        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"report_recording_names error: {error_msg}")

        # Send response
        self.send_response(
            payload=payload,
            error=error_msg,
            files=file_info_list,
        )
      
    def delete_recording(self, payload):
        """
        Delete a specific recording file.
        """
        error_msg = "None"
        recording_name = payload.get("recording_name", "")

        try:
            if not recording_name:
                raise ValueError("No file name provided for deletion.")

            path = get_data_path(self.config, recording_name, create_dirs=False)
            os.remove(path)
            self.logger_tc.info(f"Deleted recording: {recording_name}")

        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"Failed to delete recording: {error_msg}")

        # Send response
        self.send_response(
            payload=payload,
            error=error_msg,
        )

    def rename_recording(self, payload):
        """
        Rename a specific recording file (.json).
        """
        error_msg = "None"
        old_recording_name = payload.get("old_recording_name", "")
        new_recording_name = payload.get("new_recording_name", "")

        try:
            if not old_recording_name:
                raise ValueError("No old file name provided for renaming.")
            if not new_recording_name:
                raise ValueError("No new file name provided for renaming.")

            # Ensure both names have .json extension
            if not old_recording_name.endswith('.json'):
                old_recording_name += '.json'
            if not new_recording_name.endswith('.json'):
                new_recording_name += '.json'

            old_path = get_data_path(self.config, old_recording_name, create_dirs=False)
            new_path = get_data_path(self.config, new_recording_name, create_dirs=False)
            
            # Ensure the parent directory exists for the new file
            new_dir = os.path.dirname(new_path)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir, exist_ok=True)
            
            # Check if old file exists
            if not os.path.exists(old_path):
                raise ValueError(f"Recording '{old_recording_name}' does not exist.")
            
            # Check if new file already exists
            if os.path.exists(new_path):
                raise ValueError(f"Recording '{new_recording_name}' already exists.")
            
            # Rename the file (os.rename automatically deletes the old file after successful rename)
            os.rename(old_path, new_path)
            self.logger_tc.info(f"Renamed recording: {old_recording_name} -> {new_recording_name}")

        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"Failed to rename recording: {error_msg}")

        # Send response
        self.send_response(
            payload=payload,
            error=error_msg,
        )

    def rename_dataset(self, payload):
        """
        Rename a specific dataset directory.
        """
        error_msg = "None"
        old_dataset_name = payload.get("old_dataset_name", "")
        new_dataset_name = payload.get("new_dataset_name", "")

        try:
            if not old_dataset_name:
                raise ValueError("No old dataset name provided for renaming.")
            if not new_dataset_name:
                raise ValueError("No new dataset name provided for renaming.")

            # Get the data directory path
            data_path = get_data_path(self.config, create_dirs=False)
            old_path = os.path.join(data_path, old_dataset_name)
            new_path = os.path.join(data_path, new_dataset_name)
            
            # Check if old dataset directory exists
            if not os.path.exists(old_path) or not os.path.isdir(old_path):
                raise ValueError(f"Dataset '{old_dataset_name}' does not exist.")
            
            # Check if new dataset directory already exists
            if os.path.exists(new_path):
                raise ValueError(f"Dataset '{new_dataset_name}' already exists.")
            
            # Rename the directory (os.rename automatically deletes the old directory after successful rename)
            os.rename(old_path, new_path)
            self.logger_tc.info(f"Renamed dataset: {old_dataset_name} -> {new_dataset_name}")

        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"Failed to rename dataset: {error_msg}")

        # Send response
        self.send_response(
            payload=payload,
            error=error_msg,
        )

    def rename_model(self, payload):
        """
        Rename a specific model directory within a dataset.
        Models are located at: data/{dataset_name}/Models/{model_name}
        """
        error_msg = "None"
        old_model_name = payload.get("old_model_name", "")
        new_model_name = payload.get("new_model_name", "")
        dataset_name = payload.get("dataset_name", "")

        try:
            if not old_model_name:
                raise ValueError("No old model name provided for renaming.")
            if not new_model_name:
                raise ValueError("No new model name provided for renaming.")
            if not dataset_name:
                raise ValueError("No dataset name provided for model renaming.")

            # Get the Models directory path within the dataset
            models_path = get_data_path(self.config, os.path.join(dataset_name, "Models"), create_dirs=False)
            old_path = os.path.join(models_path, old_model_name)
            new_path = os.path.join(models_path, new_model_name)
            
            # Check if old model directory exists
            if not os.path.exists(old_path) or not os.path.isdir(old_path):
                raise ValueError(f"Model '{old_model_name}' does not exist in dataset '{dataset_name}'.")
            
            # Check if new model directory already exists
            if os.path.exists(new_path):
                raise ValueError(f"Model '{new_model_name}' already exists in dataset '{dataset_name}'.")
            
            # Rename the directory (os.rename automatically deletes the old directory after successful rename)
            os.rename(old_path, new_path)
            self.logger_tc.info(f"Renamed model: {old_model_name} -> {new_model_name} (dataset: {dataset_name})")

        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"Failed to rename model: {error_msg}")

        # Send response
        self.send_response(
            payload=payload,
            error=error_msg,
        )

    def delete_dataset(self, payload):
        """
        Delete a specific dataset directory and all its contents.
        """
        error_msg = "None"
        dataset_name = payload.get("dataset_name", "")

        try:
            if not dataset_name:
                raise ValueError("No dataset name provided for deletion.")

            # Get the data directory path
            data_path = get_data_path(self.config, create_dirs=False)
            dataset_path = os.path.join(data_path, dataset_name)
            
            # Check if dataset directory exists
            if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
                raise ValueError(f"Dataset '{dataset_name}' does not exist.")
            
            # Delete the entire directory and its contents
            import shutil
            shutil.rmtree(dataset_path)
            self.logger_tc.info(f"Deleted dataset: {dataset_name}")

        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"Failed to delete dataset: {error_msg}")

        # Send response
        self.send_response(
            payload=payload,
            error=error_msg,
        )

    def delete_model(self, payload):
        """
        Delete a specific model directory within a dataset.
        Models are located at: data/{dataset_name}/Models/{model_name}
        """
        error_msg = "None"
        model_name = payload.get("model_name", "")
        dataset_name = payload.get("dataset_name", "")

        try:
            if not model_name:
                raise ValueError("No model name provided for deletion.")
            if not dataset_name:
                raise ValueError("No dataset name provided for model deletion.")

            # Get the Models directory path within the dataset
            models_path = get_data_path(self.config, os.path.join(dataset_name, "Models"), create_dirs=False)
            model_path = os.path.join(models_path, model_name)
            
            # Check if model directory exists
            if not os.path.exists(model_path) or not os.path.isdir(model_path):
                raise ValueError(f"Model '{model_name}' does not exist in dataset '{dataset_name}'.")
            
            # Delete the entire directory and its contents
            import shutil
            shutil.rmtree(model_path)
            self.logger_tc.info(f"Deleted model: {model_name} (dataset: {dataset_name})")

        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"Failed to delete model: {error_msg}")

        # Send response
        self.send_response(
            payload=payload,
            error=error_msg,
        )

    def record_episodes(self, payload):
        error_msg = "None"

        try:
            if self.has_camera:
                self.connect_camera_interface()
            self.connect_save_interface()
            if self.has_teachbot:
                self.connect_teachbot_interface()
            self.connect_robot_interface()

            # Camera interface copy (only if camera is configured)
            if self.has_camera:
                camera_cmd = dict(payload)
                camera_cmd["type"] = "CMD"
                camera_cmd["interface"] = self.get_interface_name("CAMERA_INTERFACE")
                self.send_command(camera_cmd)

            # Save interface copy
            save_cmd = dict(payload)
            save_cmd["type"] = "CMD"
            save_cmd["interface"] = self.get_interface_name("SAVE_INTERFACE")
            self.send_command(save_cmd)

            # Robot interface copy
            robot_cmd = dict(payload)
            robot_cmd["type"] = "CMD"
            robot_cmd["interface"] = self.get_interface_name("ROBOT_INTERFACE")
            self.send_command(robot_cmd)

        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"record_episodes error: {error_msg}")

        self.send_response(
            payload=payload,
            error=error_msg,
        )

    def record_episode(self, payload):
        error_msg = "None"

        try:
            if self.has_camera:
                self.connect_camera_interface()
            self.connect_save_interface()
            if self.has_teachbot:
                self.connect_teachbot_interface()
            self.connect_robot_interface()

            # Camera interface copy (only if camera is configured)
            if self.has_camera:
                camera_cmd = dict(payload)
                camera_cmd["type"] = "CMD"
                camera_cmd["interface"] = self.get_interface_name("CAMERA_INTERFACE")
                self.send_command(camera_cmd)

            # Save interface copy
            save_cmd = dict(payload)
            save_cmd["type"] = "CMD"
            save_cmd["interface"] = self.get_interface_name("SAVE_INTERFACE")
            self.send_command(save_cmd)

            # Robot interface copy
            robot_cmd = dict(payload)
            robot_cmd["type"] = "CMD"
            robot_cmd["interface"] = self.get_interface_name("ROBOT_INTERFACE")
            self.send_command(robot_cmd)

        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"record_episode error: {error_msg}")

        self.send_response(
            payload=payload,
            error=error_msg,
        )

    def record_mistake(self, payload):
        """
        Move recent episodes to a _mistakes subdirectory.
        Runs in a separate thread and responds to TOS UI when complete.
        """
        error_msg = "None"
        
        try:
            # Extract dataset name from payload
            dataset_name = payload.get("dataset_name", "")
            if not dataset_name:
                error_msg = "dataset_name is required"
                self.send_response(payload=payload, error=error_msg)
                return
            
            # Start the mistake processing in a separate thread
            mistake_thread = threading.Thread(
                target=self._process_mistake_files, 
                args=(dataset_name, payload),
                daemon=True
            )
            mistake_thread.start()
            
            # Send immediate response that processing has started
            self.send_response(
                payload=payload, 
                error="None",
                status="processing_started"
            )
            
        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"record_mistake error: {error_msg}")
            self.send_response(payload=payload, error=error_msg)

    def report_dataset_names(self, payload):
        """
        Gather the list of recorded files and send them as a response,
        including the recording speed if present in the file's metadata.
        """
        error_msg = "None"
        dataset_info_list = []

        try:
            path = get_data_path(self.config)
            names = os.listdir(path)

            for object_name in names:
                full_path = os.path.join(path, object_name)

                # Only attempt to parse directories 
                if os.path.isdir(full_path):
                    dataset_info_list.append({
                        "dataset_name": object_name,
                    })

            self.logger_tc.info(f"Reported dataset names: {dataset_info_list}")

        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"report_dataset_names error: {error_msg}")

        # Send response
        self.send_response(
            payload=payload,
            error=error_msg,
            files=dataset_info_list,
        )

    def report_model_names(self, payload):
        """
        Gather the list of model directories for a specific dataset and send them as a response.
        Models are expected to be in: data/{dataset_name}/Models/{model_name}
        """
        error_msg = "None"
        model_info_list = []

        try:
            dataset_name = payload.get("dataset_name", "")

            # Construct path to Models directory within the dataset
            models_path = get_data_path(self.config, os.path.join(dataset_name, "Models"))
            
            if not os.path.exists(models_path):
                self.logger_tc.warning(f"Models directory does not exist: {models_path}")
                # Return empty list but no error
            elif not os.path.isdir(models_path):
                self.logger_tc.warning(f"Models path exists but is not a directory: {models_path}")
                # Return empty list but no error
            else:
                # List all directories in the Models folder
                model_names = os.listdir(models_path)
                
                for model_name in model_names:
                    model_full_path = os.path.join(models_path, model_name)
                    
                    # Only include directories as valid models
                    if os.path.isdir(model_full_path):
                        model_info_list.append({
                            "model_name": model_name,
                            "dataset_name": dataset_name
                        })

            self.logger_tc.info(f"Reported model names for dataset '{dataset_name}': {model_info_list}")

        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"report_model_names error: {error_msg}")

        # Send response
        self.send_response(
            payload=payload,
            error=error_msg,
            files=model_info_list,
        )

    def run_policy(self, payload):
        error_msg = "None"

        try:
            if self.has_camera:
                self.connect_camera_interface()
            if self.has_policy:
                self.connect_policy_interface()
            self.connect_robot_interface()

            # Camera interface copy (only if camera is configured)
            if self.has_camera:
                camera_cmd = dict(payload)
                camera_cmd["type"] = "CMD"
                camera_cmd["interface"] = self.get_interface_name("CAMERA_INTERFACE")
                self.send_command(camera_cmd)

            # Policy interface copy (only if policy is configured)
            if self.has_policy:
                policy_cmd = dict(payload)
                policy_cmd["type"] = "CMD"
                policy_cmd["interface"] = self.get_interface_name("POLICY_INTERFACE")
                self.send_command(policy_cmd)

            # Robot interface copy
            robot_cmd = dict(payload)
            robot_cmd["type"] = "CMD"
            robot_cmd["interface"] = self.get_interface_name("ROBOT_INTERFACE")
            self.send_command(robot_cmd)

        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"run_policy error: {error_msg}")

        self.send_response(
            payload=payload,
            error=error_msg,
        )        

    def train_policy(self, payload):
        """
        Connect the necessary interfaces and start training a policy.
        """
        error_msg = "None"

        try:
            self.connect_policy_interface()

            # Policy interface copy
            policy_cmd = dict(payload)
            policy_cmd["type"] = "CMD"
            policy_cmd["interface"] = self.get_interface_name("POLICY_INTERFACE")
            self.send_command(policy_cmd)

        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"train_policy error: {error_msg}")

        # Send response
        self.send_response(
            payload=payload,
            error=error_msg,
        )


    def stop(self, payload):
        """
        Disconnect interfaces but keep the controller up.
        """
        error_msg = "None"
        try:
            self.disconnect_interfaces_only()
            self.logger_tc.info("Interfaces stopped, controller remains active.")
        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"stop error: {error_msg}")

        # Send response
        self.send_response(
            payload=payload,
            error=error_msg,
        )
        return True

    def hard_stop(self, payload):
        """
        Full stop. Disconnect everything and shut down the controller.
        """
        error_msg = "None"
        try:
            self.logger_tc.info("Performing HARD STOP. Everything shutting down.")
            self.disconnect_all()
        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"hard_stop error: {error_msg}")

        # Send response
        self.send_response(
            payload=payload,
            error=error_msg,
        )


    ###################################################################
    # 1) Parse config & load relevant sections
    ###################################################################
    def _parse_config(self):
        try:
            # If setup_id is provided, use setup-specific configuration
            if self.setup_id:
                setup_name = f"setup_{self.setup_id}"
                self.logger_tc.info(f"Loading configuration for {setup_name}")
                
                if setup_name not in self.config["hardware"]:
                    self.logger_tc.error(f"Setup {setup_name} not found in hardware configuration")
                    sys.exit(1)
                
                hw = self.config["hardware"][setup_name]
                
                # Store the numerical setup_id for RabbitMQ routing
                self.setup_routing_id = self.config["hardware"][setup_name].get("setup_id", self.setup_id)
            else:
                # Fallback to global hardware config (legacy mode)
                self.logger_tc.warning("No setup_id provided, using global hardware configuration")
                hw = self.config["hardware"]
                self.setup_routing_id = None
            
            self.robot_brand = hw["robot"]["brand"].lower()
            
            # Teachbot is optional - set to None if not configured
            if "teachbot" in hw and "brand" in hw["teachbot"]:
                self.teachbot_brand = hw["teachbot"]["brand"].lower()
                self.has_teachbot = True
            else:
                self.teachbot_brand = None
                self.has_teachbot = False
                self.logger_tc.info("No teachbot configuration found, teachbot functionality disabled")
                
            # Camera is optional - set to None if not configured
            if "camera" in hw and "brand" in hw["camera"]:
                self.camera_brand = hw["camera"]["brand"].lower()
                self.has_camera = True
            else:
                self.camera_brand = None
                self.has_camera = False
                self.logger_tc.info("No camera configuration found, camera functionality disabled")
            self.control_dt = hw["robot"]["control_dt"]
            # Policy is optional - set to None if not configured  
            if "policy" in self.config and "name" in self.config["policy"]:
                self.policy_name = self.config["policy"]["name"].lower()
                self.has_policy = True
            else:
                self.policy_name = None
                self.has_policy = False
                self.logger_tc.info("No policy configuration found, policy functionality disabled")
            
            # Store the hardware config for use throughout the class
            self.hw_config = hw
            
            # Create a modified config for modules that need full config structure
            # but with setup-specific hardware configuration
            self.effective_config = self.config.copy()
            if self.setup_id:
                self.effective_config["hardware"] = hw   

        except KeyError as e:
            self.logger_tc.error(f"Missing hardware config: {e}")
            sys.exit(1)

        try:
            rbmq = self.config["rabbitmq"]
            self.RABBIT_HOST = rbmq["host"]
            self.RABBIT_USER = rbmq["user"]
            self.RABBIT_PASS = rbmq["pass"]
            self.RABBIT_HEARTBEAT = rbmq["heartbeat"]
            self.RABBIT_BLOCKED_CONNECTION_TIMEOUT = rbmq["blocked_connection_timeout"]
            self.RABBIT_MAX_RETRIES = rbmq["max_retries"]
            self.RABBIT_WAIT_SECONDS = rbmq["wait_seconds"]

            # client props
            self.RABBIT_CLIENT_NAME_DEFAULT = rbmq["client_name_default"]
            self.RABBIT_PRODUCT = rbmq["product"]
            self.RABBIT_INFORMATION = rbmq["information"]

            # exchange, queue, binding
            self.EXCHANGE_NAME = rbmq["exchange_name"]
            self.EXCHANGE_TYPE = rbmq["exchange_type"]
            
            # Setup-specific queue and routing configuration
            if self.setup_routing_id:
                # Multi-setup mode: use new pattern-based configuration
                patterns = rbmq["robot_queue_patterns"]
                self.COMMAND_QUEUE_NAME = patterns["command_queue"].format(setup_id=self.setup_routing_id)
                self.COMMAND_BINDING_KEY = f"robot_controller.{self.setup_routing_id}.command.*"
                self.RESPONSE_ROUTING_KEY_BASE = patterns["response_routing_key"]  # Store pattern with {action} placeholder
                self.STATUS_ROUTING_KEY = patterns["status_routing_key"].format(setup_id=self.setup_routing_id)
                self.logger_tc.info(f"Using setup-specific routing: queue={self.COMMAND_QUEUE_NAME}, binding={self.COMMAND_BINDING_KEY}")
            else:
                # Legacy mode: use old configuration if available
                if "command_queue_name" in rbmq:
                    self.COMMAND_QUEUE_NAME = rbmq["command_queue_name"]
                    prefix = rbmq["command_binding_key_prefix"]  # e.g. "robot_controller.command."
                    self.COMMAND_BINDING_KEY = prefix + "#"
                    self.RESPONSE_ROUTING_KEY_BASE = rbmq["response_binding_key_prefix"] + ".{action}"  # Legacy format with action placeholder
                    self.STATUS_ROUTING_KEY = rbmq["status_binding_key"]
                else:
                    # Use default fallback for single setup
                    patterns = rbmq["robot_queue_patterns"]
                    self.COMMAND_QUEUE_NAME = patterns["command_queue"].format(setup_id="1")
                    self.COMMAND_BINDING_KEY = "robot_controller.1.command.*"
                    self.RESPONSE_ROUTING_KEY_BASE = patterns["response_routing_key"]  # Store pattern with {action} placeholder
                    self.STATUS_ROUTING_KEY = patterns["status_routing_key"].format(setup_id="1")
        except KeyError as e:
            self.logger_tc.error(f"Missing RabbitMQ config: {e}")
            sys.exit(1)

        gen = self.config["general"]
        self.check_queue_period = gen["check_queue_period"]
        self.status_refresh_period = gen["status_refresh_period"]
        self.app_directory = gen["app_directory"]
        self.data_directory = gen["data_directory"]
        self.control_loop_language = gen["control_loop_language"]
        self.record_duration = gen.get("record_duration", 1)
        self.policy_img_buffer_size = gen.get("policy_img_buffer_size", 1)

        # Policy configuration (optional)
        if self.has_policy:
            try:
                policy_config = self.config["policy"]
                self.shm_target_pos2_config = policy_config["shared_memory"]
            except KeyError as e:
                self.logger_tc.error(f"Missing policy config: {e}")
                sys.exit(1)
        else:
            self.shm_target_pos2_config = None
            self.logger_tc.info("Policy configuration disabled, skipping policy shared memory setup")

        # General run states
        self.processes_running = True
        self.cleanup_done = False

    def _load_dynamic_modules(self):
        """
        Dynamically import robot and teachbot brand-specific modules
        for their respective interfaces.
        """
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

        # Robot interface module
        try:
            robot_module_name = f"modules.robot.{self.robot_brand}.robot_{self.robot_brand}"
            robot_module = importlib.import_module(robot_module_name)
            self.run_robot_interface = robot_module.run_robot_interface
            self.logger_tc.info(f"Imported {robot_module_name} for robot_brand={self.robot_brand}")
        except ImportError as e:
            self.logger_tc.error(f"Unsupported robot brand: {self.robot_brand} => {e}")
            sys.exit(1)

        # Teachbot interface module (only if teachbot is configured)
        if self.has_teachbot:
            try:
                teachbot_module_name = f"modules.teachbot.teachbot_{self.teachbot_brand}"
                teachbot_module = importlib.import_module(teachbot_module_name)
                self.run_teachbot_interface = teachbot_module.run_teachbot_interface
                self.logger_tc.info(f"Imported {teachbot_module_name} for teachbot_brand={self.teachbot_brand}")
            except ImportError as e:
                self.logger_tc.error(f"Unsupported teachbot brand: {self.teachbot_brand} => {e}")
                sys.exit(1)
        else:
            self.run_teachbot_interface = None
            self.logger_tc.warning("Teachbot interface disabled - no teachbot configured")

        # Save interface is known
        from modules.save.save_interface import run_save_interface
        self.run_save_interface = run_save_interface

        # Camera interface module (only if camera is configured)
        if self.has_camera:
            try:
                camera_module_name = f"modules.camera.camera_{self.camera_brand.lower()}"
                camera_module = importlib.import_module(camera_module_name)
                self.run_camera_interface = camera_module.run_camera_interface
                self.logger_tc.info(f"Imported {camera_module_name} for camera_brand={self.camera_brand}")
            except ImportError as e:
                self.logger_tc.error(f"Unsupported camera brand: {self.camera_brand} => {e}")
                sys.exit(1)
        else:
            self.run_camera_interface = None
            self.logger_tc.info("Camera interface disabled - no camera configured")

        # Policy interface module (only if policy is configured)
        if self.has_policy:
            try:
                policy_module_name = f"modules.policy.{self.policy_name}.policy_{self.policy_name}"
                self.logger_tc.info(f"Importing policy module: {policy_module_name}")
                policy_module = importlib.import_module(policy_module_name)
                self.run_policy_interface = policy_module.run_policy_interface
                self.logger_tc.info(f"Imported {policy_module_name} for policy_type={self.policy_name}")
            except ImportError as e:
                self.logger_tc.error(f"Unsupported policy type: {self.policy_name} => {e}")
                sys.exit(1)
        else:
            self.run_policy_interface = None
            self.logger_tc.info("Policy interface disabled - no policy configured")

    def _init_interface_states(self):
        """
        Set up multiprocessing queues, placeholders, flags, etc.
        """
        # Interface states
        self.teachbot_interface_occupied    = False
        self.robot_interface_occupied       = False
        self.save_interface_occupied        = False
        self.camera_interface_occupied       = False
        self.policy_interface_occupied       = False

        # Multiprocessing Queues
        # Teachbot communication
        self.teachbot_interface_commup   = multiprocessing.Queue(maxsize=100)
        self.teachbot_interface_commdown = multiprocessing.Queue(maxsize=1000)

        # Robot communication
        self.robot_interface_commup   = multiprocessing.Queue(maxsize=100)
        self.robot_interface_commdown = multiprocessing.Queue(maxsize=1000)

        # Save communication
        self.save_interface_commup   = multiprocessing.Queue(maxsize=100)
        self.save_interface_commdown = multiprocessing.Queue(maxsize=1000)

        # Camera communication - separate queues per camera
        self.camera_interface_commup = multiprocessing.Queue(maxsize=100)
        self.camera_interface_commdown = {}  # Will be dict of {camera_name: queue}

        # Policy communication
        self.policy_interface_commup   = multiprocessing.Queue(maxsize=100)
        self.policy_interface_commdown = multiprocessing.Queue(maxsize=1000)


        # Teachbot data
        self.shm_target_pos1 = multiprocessing.Queue(maxsize=1)

        # Robot data
        if self.control_loop_language == "python":
            size = int(self.record_duration / self.control_dt)
            self.shm_joint_data1 = multiprocessing.Queue(maxsize=size)
            self.shm_joint_data2 = multiprocessing.Queue(maxsize=size)

        elif self.control_loop_language == "cpp":
            # Initialize shared memory for C++ control loop
            from utils.utils import calculate_shm_capacity
            size = calculate_shm_capacity(self.record_duration, self.control_dt)
            self.initialize_shared_memory(size) # cpp_shm_joint_data1 and cpp_shm_joint_data2
            self.shm_joint_data1 = None
            self.shm_joint_data2 = None

        # Camera data
        self.initialize_camera_shared_memory() # color_buffers1, depth_buffers1, color_buffers2, depth_buffers2

        # Policy data
        self.initialize_shm_target_pos2()


        # Process placeholders
        self.teachbot_interface_process = None 
        self.robot_interface_process = None
        self.save_interface_process  = None
        self.camera_processes = {}  # Changed from list to dict for camera name mapping
        self.policy_interface_process = None

        self.commup_thread_teachbot_interface = None
        self.commup_thread_robot_interface    = None
        self.commup_thread_save_interface     = None
        self.commup_thread_camera_interface   = None
        self.commup_thread_policy_interface   = None

        self.status_pub_tc_thread            = None
        self.command_consumer_thread         = None

        # For controlling watchers
        self.checking_status = False


    ###################################################################
    # Shared Memory Management for C++ Interface and cameras
    ###################################################################
    
    def initialize_shared_memory(self, capacity):
        """
        Initialize shared memory for C++ control loop.
        Creates two shared memory segments:
        - shm_cpp_joint_data1: for save interface (recording)
        - shm_cpp_joint_data2: for policy interface (real-time control)
        """
        if self.control_loop_language != "cpp":
            return
            
        self.logger_tc.info("Using C++ shared memory interface for recording and policy.")
        
        # Read constants from config
        DOF_ROBOT = self.hw_config["robot"]["dof"]  # Robot joints only
        DOF_EE = self.hw_config["robot"]["dof_ee"]   # End effector DOF (gripper)  
        DOF = DOF_ROBOT + DOF_EE  # Total DOF including gripper
        
        # Initialize both shared memory segments
        self._initialize_single_shared_memory("shm_joint_data1", capacity, DOF)
        # Policy interface shared memory only if policy is configured
        if self.has_policy:
            # Use deterministic calculation for policy interface too (smaller buffer for real-time data)
            policy_capacity = self.config["cpp"]["shared_memory"]["shm_joint_data2"]["capacity"]
            self._initialize_single_shared_memory("shm_joint_data2", policy_capacity, DOF)
        
        # Initialize the readers with effective config (setup-specific hardware + global other sections)
        self.shm_reader1 = RingBufferReader(self.effective_config, "shm_joint_data1", setup_id=str(self.setup_id))
        if self.has_policy:
            self.shm_reader2 = RingBufferReader(self.effective_config, "shm_joint_data2", setup_id=str(self.setup_id))
        else:
            self.shm_reader2 = None
        return self.shm_reader1, self.shm_reader2
        
    def _initialize_single_shared_memory(self, shm_key, capacity, dof):
        """
        Initialize a single shared memory segment.
        """
        cpp_config = self.config["cpp"]["shared_memory"][shm_key]
        CAPACITY = capacity
        SLOT_FMT_TEMPLATE = cpp_config["slot_format_template"]
        SLOT_FMT = SLOT_FMT_TEMPLATE.format(dof=dof)  # Dynamically substitute total DOF
        SLOT_SIZE = struct.calcsize(SLOT_FMT)
        HEADER_FMT = cpp_config["header_format"]
        HEADER_SIZE = struct.calcsize(HEADER_FMT)
        # Make shared memory name setup-specific
        SHM_NAME = f"{int(self.setup_id):02d}_{cpp_config['shm_name']}"
        
        self.logger_tc.info("Creating shared memory segment: %s with DOF=%d", SHM_NAME, dof)
        self.logger_tc.info("SLOT_FMT_TEMPLATE: %s", SLOT_FMT_TEMPLATE)
        self.logger_tc.info("SLOT_FMT: %s", SLOT_FMT)
        self.logger_tc.info("SLOT_SIZE: %d bytes", SLOT_SIZE)
        self.logger_tc.info("HEADER_SIZE: %d bytes", HEADER_SIZE)
        self.logger_tc.info("CAPACITY: %d slots", CAPACITY)
        bytes_needed = HEADER_SIZE + CAPACITY * SLOT_SIZE
        self.logger_tc.info("Total bytes needed: %d", bytes_needed)
        
        # Try to create shared memory, handling the case where it already exists
        try:
            shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=bytes_needed)
            # initialise header once
            struct.pack_into(HEADER_FMT, shm.buf, 0, 0, 0, CAPACITY, SLOT_SIZE)
            self.logger_tc.info("Shared memory segment created with size: %d bytes", bytes_needed)
            shm.close()
        except FileExistsError:
            self.logger_tc.warning("Shared memory segment already exists, cleaning up and recreating")
            try:
                # Try to attach to existing memory and clean it up
                existing_shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
                existing_shm.close()
                existing_shm.unlink()
                self.logger_tc.info("Cleaned up existing shared memory segment")
                
                # Now create new shared memory
                shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=bytes_needed)
                struct.pack_into(HEADER_FMT, shm.buf, 0, 0, 0, CAPACITY, SLOT_SIZE)
                self.logger_tc.info("Shared memory segment recreated with size: %d bytes", bytes_needed)
                shm.close()
            except Exception as cleanup_error:
                self.logger_tc.error("Failed to clean up existing shared memory: %s", cleanup_error)
                raise
        
    def cleanup_shared_memory_cpp(self):
        """
        Clean up shared memory for C++ control loop.
        """
        if self.control_loop_language != "cpp":
            return
            
        # Clean up shared memory segments
        shm_pairs = [("shm_joint_data1", "shm_reader1")]
        if self.has_policy:
            shm_pairs.append(("shm_joint_data2", "shm_reader2"))
            
        for shm_key, reader_attr in shm_pairs:
            if hasattr(self, reader_attr):
                try:
                    # First close the reader to release memory views
                    reader = getattr(self, reader_attr)
                    if reader and hasattr(reader, 'close'):
                        reader.close()
                        self.logger_tc.info(f"Closed RingBufferReader for {shm_key}")
                    
                    # Then clean up the shared memory segment
                    shm_name = self.config["cpp"]["shared_memory"][shm_key]["shm_name"]
                    try:
                        shm = shared_memory.SharedMemory(name=shm_name, create=False)
                        shm.close()
                        shm.unlink()  # Remove the shared memory segment
                        self.logger_tc.info(f"C++ shared memory segment {shm_name} cleaned up.")
                    except FileNotFoundError:
                        self.logger_tc.info(f"C++ shared memory segment {shm_name} already cleaned up.")
                except Exception as e:
                    self.logger_tc.error(f"Failed to clean up C++ shared memory {shm_key}: %s", e)

    def get_shm_cpp_joint_data2_info(self):
        """
        Get shm_cpp_joint_data2 shared memory information for policy interface when using C++ control loop.
        Returns a dict with all necessary parameters to access the shared memory.
        """
        if self.control_loop_language != "cpp" or not self.has_policy:
            return None
            
        DOF_ROBOT = self.hw_config["robot"]["dof"]  # Robot joints only
        DOF_EE = self.hw_config["robot"]["dof_ee"]   # End effector DOF (gripper)
        DOF = DOF_ROBOT + DOF_EE  # Total DOF including gripper
        
        cpp_config = self.config["cpp"]["shared_memory"]["shm_joint_data2"]
        SLOT_FMT_TEMPLATE = cpp_config["slot_format_template"]
        SLOT_FMT = SLOT_FMT_TEMPLATE.format(dof=DOF)
        HEADER_FMT = cpp_config["header_format"]
        
        return {
            'name': f"{int(self.setup_id):02d}_{cpp_config['shm_name']}",
            'capacity': cpp_config["capacity"],
            'slot_format': SLOT_FMT,
            'slot_size': struct.calcsize(SLOT_FMT),
            'header_format': HEADER_FMT,
            'header_size': struct.calcsize(HEADER_FMT),
            'dof': DOF
        }

    def initialize_camera_shared_memory(self):
        if not self.has_camera:
            self.logger_tc.info("Camera functionality disabled, skipping camera shared memory initialization.")
            self.camera_cfgs = []
            # Initialize camera buffer attributes as None when cameras are disabled
            self.color_buffers1 = None
            self.depth_buffers1 = None
            self.color_buffers2 = None
            self.depth_buffers2 = None
            return
            
        try:
            self.camera_cfgs = self.hw_config["camera"]["info"]
            # self.camera_fps  = self.camera_cfgs["fps"][0]
        except KeyError as e:
            self.logger_tc.info(f"Camera section missing in config: {e}")
            self.camera_cfgs = []

        # Initialize ring buffer manager and create buffers for each camera
        self.logger_tc.info("Initializing camera ring buffers.")
        
        self.camera_ring_buffer_manager = CameraRingBufferManager(setup_id=str(self.setup_id))
        buffers_info = self.camera_ring_buffer_manager.create_buffers(
            self.camera_cfgs, 
            record_duration=self.record_duration,
            policy_img_buffer_size=self.policy_img_buffer_size
        )
        
        # Store ring buffer references
        self.color_buffers1 = buffers_info["color_buffers1"]
        self.depth_buffers1 = buffers_info["depth_buffers1"]
        self.color_buffers2 = buffers_info["color_buffers2"]
        self.depth_buffers2 = buffers_info["depth_buffers2"]
        
        for camera in self.camera_cfgs:
            name = camera["name"]
            fps = camera["fps"]
            sn = camera["serial_number"]
            cw,ch = camera["color_width"],  camera["color_height"]
            dw,dh = camera["depth_width"],  camera["depth_height"]
            
            # Create separate command queue for each camera (keep this as queue)
            self.camera_interface_commdown[name] = multiprocessing.Queue(maxsize=1000)

            # Get buffer capacities for logging
            color_buffer1 = self.color_buffers1[name]
            color_buffer2 = self.color_buffers2[name]
            recording_capacity = color_buffer1.capacity
            policy_capacity = color_buffer2.capacity
            self.logger_tc.info(
                f"Initialized ring buffers for camera '{name}' (SN: {sn}) "
                f"Color: {cw}x{ch}, Depth: {dw}x{dh}, "
                f"Recording buffer capacity: {recording_capacity} frames, "
                f"Policy buffer capacity: {policy_capacity} frames."
            )

    def cleanup_shared_memory_cameras(self):
        """
        Clean up shared memory for camera ring buffers.
        """
        self.logger_tc.info("Cleaning up camera shared memory...")
        
        # Clean up camera ring buffer manager
        if hasattr(self, 'camera_ring_buffer_manager') and self.camera_ring_buffer_manager:
            try:
                self.camera_ring_buffer_manager.close_all(unlink=True)
                self.logger_tc.info("Camera ring buffers cleaned up successfully.")
            except Exception as e:
                self.logger_tc.error(f"Error cleaning up camera ring buffers: {e}")
        
        # Clean up individual buffer references
        for attr_name in ['color_buffers1', 'depth_buffers1', 'color_buffers2', 'depth_buffers2']:
            if hasattr(self, attr_name):
                buffers = getattr(self, attr_name)
                if isinstance(buffers, dict):
                    for camera_name, buffer_obj in buffers.items():
                        try:
                            if hasattr(buffer_obj, 'close'):
                                buffer_obj.close(unlink=False)  # Don't unlink here as manager already did it
                        except Exception as e:
                            self.logger_tc.error(f"Error closing {attr_name} buffer for {camera_name}: {e}")
                
                # Clear the attribute
                setattr(self, attr_name, {})
        
        # Reset camera ring buffer manager
        if hasattr(self, 'camera_ring_buffer_manager'):
            self.camera_ring_buffer_manager = None
            
        self.logger_tc.info("Camera shared memory cleanup complete.")
        

        # Terminate all camera processes
        for camera_name, process in self.camera_processes.items():
            if process and process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
                if process.is_alive():
                    process.kill()
                    process.join()
                self.logger_tc.info(f"Camera process for '{camera_name}' terminated.")
        

        # Clean up ring buffers
        if hasattr(self, 'camera_ring_buffer_manager') and self.camera_ring_buffer_manager:
            try:
                self.camera_ring_buffer_manager.close_all(unlink=True)
                self.logger_tc.info("Camera ring buffers cleaned up.")
            except Exception as e:
                self.logger_tc.error(f"Error cleaning up camera ring buffers: {e}")


    def initialize_shm_target_pos2(self):
        """
        Initialize shared memory for policy interface using config values.
        Structure: sequence_id (uint32) + joint_positions (DOF doubles)
        """
        if not self.has_policy:
            self.logger_tc.info("Policy functionality disabled, skipping policy shared memory initialization.")
            return
            
        # Read configuration
        DOF_ROBOT = self.hw_config["robot"]["dof"]  # Robot joints only
        DOF_EE = self.hw_config["robot"]["dof_ee"]   # End effector DOF (gripper)
        DOF = DOF_ROBOT + DOF_EE  # Total DOF including gripper
        # Make shared memory name setup-specific
        SHM_NAME = f"{int(self.setup_id):02d}_{self.shm_target_pos2_config['shm_name']}"
        CAPACITY = self.shm_target_pos2_config["capacity"]
        ENTRY_FMT_TEMPLATE = self.shm_target_pos2_config["entry_format_template"]
        
        # Create the actual format string by substituting total DOF
        ENTRY_FMT = ENTRY_FMT_TEMPLATE.format(dof=DOF)
        ENTRY_SIZE = struct.calcsize(ENTRY_FMT)
        
        self.logger_tc.info("Creating policy shared memory segment: %s with DOF=%d, capacity=%d", SHM_NAME, DOF, CAPACITY)
        bytes_needed = CAPACITY * ENTRY_SIZE
        
        # Try to create shared memory, handling the case where it already exists
        try:
            shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=bytes_needed)
            # Initialize all entries to zero
            shm.buf[:] = b'\x00' * bytes_needed
            self.logger_tc.info("Policy shared memory segment created with size: %d bytes (%d entries of %d bytes each)", 
                               bytes_needed, CAPACITY, ENTRY_SIZE)
            shm.close()
        except FileExistsError:
            self.logger_tc.warning("Policy shared memory segment already exists, cleaning up and recreating")
            try:
                # Try to attach to existing memory and clean it up
                existing_shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
                existing_shm.close()
                existing_shm.unlink()
                self.logger_tc.info("Cleaned up existing policy shared memory segment")
                
                # Now create new shared memory
                shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=bytes_needed)
                shm.buf[:] = b'\x00' * bytes_needed
                self.logger_tc.info("Policy shared memory segment recreated with size: %d bytes", bytes_needed)
                shm.close()
            except Exception as cleanup_error:
                self.logger_tc.error("Failed to clean up existing policy shared memory: %s", cleanup_error)
                raise
        
        self.logger_tc.info("Policy shared memory initialized successfully. Entry format: %s", ENTRY_FMT)

    def get_shm_target_pos2_info(self):
        """
        Get shm_target_pos2 shared memory information for interfaces.
        Returns a dict with all necessary parameters to access the shared memory.
        """
        if not self.has_policy:
            return None
            
        DOF_ROBOT = self.hw_config["robot"]["dof"]  # Robot joints only
        DOF_EE = self.hw_config["robot"]["dof_ee"]   # End effector DOF (gripper)
        DOF = DOF_ROBOT + DOF_EE  # Total DOF including gripper
        ENTRY_FMT_TEMPLATE = self.shm_target_pos2_config["entry_format_template"]
        ENTRY_FMT = ENTRY_FMT_TEMPLATE.format(dof=DOF)
        
        return {
            'name': f"{int(self.setup_id):02d}_{self.shm_target_pos2_config['shm_name']}",
            'capacity': self.shm_target_pos2_config["capacity"],
            'entry_format': ENTRY_FMT,
            'entry_size': struct.calcsize(ENTRY_FMT),
            'dof': DOF
        }
        
    def cleanup_shm_target_pos2(self):
        """
        Clean up shared memory for policy interface.
        """
        if not self.has_policy:
            self.logger_tc.info("Policy functionality disabled, skipping policy shared memory cleanup.")
            return
            
        try:
            # Use setup-specific name for cleanup
            SHM_NAME = f"{int(self.setup_id):02d}_{self.shm_target_pos2_config['shm_name']}"
            shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
            shm.close()
            shm.unlink()  # Remove the shared memory segment
            self.logger_tc.info("Policy shared memory segment cleaned up.")
        except Exception as e:
            self.logger_tc.error("Failed to clean up policy shared memory: %s", e)


    ###################################################################
    # 2) RabbitMQ Setup
    ###################################################################
    def setup_rabbitmq_infrastructure(self):
        """
        Declare exchange + the controller command queue & binding once.
        """
        self.logger_tc.info("Setting up RabbitMQ infrastructure (RobotController).")
        with open_channel(self._rabbit_conf_dict(), self.logger_tc, "teachbot_infra_setup") as channel:
            # Exchange
            channel.exchange_declare(
                exchange=self.EXCHANGE_NAME,
                exchange_type=self.EXCHANGE_TYPE,
                durable=False,
            )
            # Command queue + bind
            channel.queue_declare(queue=self.COMMAND_QUEUE_NAME, durable=False)
            channel.queue_bind(
                exchange=self.EXCHANGE_NAME,
                queue=self.COMMAND_QUEUE_NAME,
                routing_key=self.COMMAND_BINDING_KEY
            )
        self.logger_tc.info("Infrastructure setup complete for command queue.")

    def _rabbit_conf_dict(self):
        """
        Helper to produce a dict that can be used with open_channel / robust_connect.
        """
        return {
            "host": self.RABBIT_HOST,
            "user": self.RABBIT_USER,
            "pass": self.RABBIT_PASS,
            "heartbeat": self.RABBIT_HEARTBEAT,
            "blocked_connection_timeout": self.RABBIT_BLOCKED_CONNECTION_TIMEOUT,
            "max_retries": self.RABBIT_MAX_RETRIES,
            "wait_seconds": self.RABBIT_WAIT_SECONDS,
            "client_name_default": self.RABBIT_CLIENT_NAME_DEFAULT,
            "product": self.RABBIT_PRODUCT,
            "information": self.RABBIT_INFORMATION,
            "exchange_name": self.EXCHANGE_NAME,
            "exchange_type": self.EXCHANGE_TYPE
        }


    ###################################################################
    # 3) Start/Stop Consumer
    ###################################################################
    def start_command_consumer(self):
        """
        Launch the background thread that consumes commands from the
        "controller.command.#" queue.
        """
        self.logger_tc.info("Starting command consumer thread.")
        self.command_consumer_thread = threading.Thread(
            target=robust_consume,
            args=(
                self._rabbit_conf_dict(),
                self.logger_tc,
                self.COMMAND_QUEUE_NAME,
                self.COMMAND_BINDING_KEY,
                self.on_command_message,
                lambda: not self.processes_running  # stop if processes_running == False
            ),
            daemon=True
        )
        self.command_consumer_thread.start()

    def on_command_message(self, ch, method, properties, body):
        """
        Callback for commands from 'controller.command.#'.
        We call self.<command_type>(payload). If no method, log error.
        """
        payload = {}
        command_type = ""

        try:
            payload = json.loads(body.decode("utf-8"))
            self.logger_tc.info(f"Received command: {payload}")
            command_type = payload.get("message", "")
        except Exception as e:
            self.logger_tc.warning(f"Skipping invalid message: {body}, error={e}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        try:
            command_func = getattr(self, command_type)
            command_func(payload)
        except AttributeError:
            self.logger_tc.error(f"Unknown command/message: {command_type}")
        except TypeError as e:
            self.logger_tc.error(f"Method '{command_type}' found but call failed: {e}")
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)


    ###################################################################
    # 4) Response Publishing
    ###################################################################
    def send_response_message(self, command, payload):
        """
        Publish a response message to e.g. "robot_controller.1.response.<command>".
        The 'payload' is already the final dict that includes "type":"RESP", etc.
        """
        routing_key = self.RESPONSE_ROUTING_KEY_BASE.format(setup_id=self.setup_routing_id, action=command)
        publish_message(
            self._rabbit_conf_dict(),
            self.logger_tc,
            routing_key=routing_key,
            message=json.dumps(payload),
            client_name="robot_controller_response"
        )
        self.logger_tc.info(f"Sent response on key={routing_key}: {payload}")

    def send_response(self, payload, error="None", **kwargs):
        """
        Helper to construct and send the new response dict:
          {
            "type": "RESP",
            "message": <e.g. "start_teleoperation">,
            "error": "<any error string>",
            ... plus any extra fields from kwargs ...
          }
        Then calls send_response_message(...)
        """
        message = payload.get("message")

        response = payload.copy()
        response["type"] = "RESP"
        if error not in ("None", ""):
            response["error"] = error
        elif response.get("error", "") == "":
            response["error"] = "None"

        # Add additional fields if present
        response.update(kwargs)  # merges any additional fields (like files, recording_name, etc.)

        # Log and publish
        self.logger_tc.info(f"Preparing to send response: {response}")
        self.send_response_message(message, response)


    ###################################################################
    # 5) Publish Commands Internally
    ###################################################################
    def publish_controller_command(self, cmd):
        """
        Publish a command back into the "controller.command.#" queue
        (like a self-trigger, or triggered from an interface event).
        """
        if self.setup_routing_id:
            # Multi-setup mode: use setup-specific routing
            routing_key = f"robot_controller.{self.setup_routing_id}.command.from_rc"
        else:
            # Legacy mode or single setup: try old config first, then fallback
            if hasattr(self, 'RESPONSE_ROUTING_KEY_BASE') and 'command_binding_key_prefix' in self.config.get('rabbitmq', {}):
                routing_key = f"{self.config['rabbitmq']['command_binding_key_prefix']}from_rc"
            else:
                self.logger_tc.warning("No setup-specific routing found, using default 'robot_controller.command.from_rc'")
        
        cmd_json = json.dumps(cmd)
        publish_message(
            self._rabbit_conf_dict(),
            self.logger_tc,
            routing_key=routing_key,
            message=cmd_json,
            client_name="robot_controller_cmd_producer"
        )
        self.logger_tc.info(f"Published command: {cmd}")

    def send_command(self, command):
        """
        Send a command to one of the local processes:
          TEACHBOT_INTERFACE, ROBOT_INTERFACE, SAVE_INTERFACE
        Waits until the process is "accepting commands", then pushes it.
        """
        self.logger_tc.info(f"Waiting to send local command: {command}")
        interface_name = command.get("interface", "")
        message = command.get("message", "")

        if not interface_name:
            self.logger_tc.error("No 'interface' in command. Ignoring.")
            return

        # Extract base interface name for routing
        base_interface_name = self.get_base_interface_name(interface_name)

        if base_interface_name == "TEACHBOT_INTERFACE":
            if not self.has_teachbot:
                self.logger_tc.error("TEACHBOT_INTERFACE requested but teachbot not configured.")
                return
            if self.teachbot_interface_process is None:
                self.logger_tc.warning("TEACHBOT_INTERFACE process not running.")
                return
            while self.teachbot_interface_occupied is not False:
                time.sleep(self.status_refresh_period)
            # Mark it as 'occupied' with the message to wait for
            self.teachbot_interface_occupied = message
            self.logger_tc.info(f"TEACHBOT_INTERFACE occupied with: {message}")
            self.teachbot_interface_commdown.put(command)
            self.logger_tc.info(f"Sent local command: {command}")

        elif base_interface_name == "ROBOT_INTERFACE":
            if self.robot_interface_process is None:
                self.logger_tc.warning("ROBOT_INTERFACE process not running.")
                return
            while self.robot_interface_occupied is not False:
                time.sleep(self.status_refresh_period)
            self.robot_interface_occupied = message
            self.logger_tc.info(f"ROBOT_INTERFACE occupied with: {message}")
            self.robot_interface_commdown.put(command)
            self.logger_tc.info(f"Sent local command: {command}")

        elif base_interface_name == "SAVE_INTERFACE":
            if self.save_interface_process is None:
                self.logger_tc.warning("SAVE_INTERFACE process not running.")
                return
            while self.save_interface_occupied is not False:
                time.sleep(self.status_refresh_period)
            self.save_interface_occupied = message
            self.logger_tc.info(f"SAVE_INTERFACE occupied with: {message}")
            self.save_interface_commdown.put(command)
            self.logger_tc.info(f"Sent local command: {command}")

        elif base_interface_name == "CAMERA_INTERFACE":
            if not hasattr(self, 'camera_processes') or not self.camera_processes:
                self.logger_tc.warning("CAMERA_INTERFACE processes not running.")
                return
            
            # Wait for camera interface to be available
            while self.camera_interface_occupied is not False:
                time.sleep(self.status_refresh_period)
            
            # Mark camera interface as occupied
            self.camera_interface_occupied = message
            self.logger_tc.info(f"CAMERA_INTERFACE occupied with: {message}")
            
            # Send command to each camera's individual queue - NO RACE CONDITIONS!
            for camera_name in self.camera_processes.keys():
                self.camera_interface_commdown[camera_name].put(command)
            
            self.logger_tc.info(f"Sent command to all {len(self.camera_processes)} camera interfaces: {command}")

        elif base_interface_name == "POLICY_INTERFACE":
            if self.policy_interface_process is None:
                self.logger_tc.warning("POLICY_INTERFACE process not running.")
                return
            while self.policy_interface_occupied is not False:
                time.sleep(self.status_refresh_period)
            self.policy_interface_occupied = message
            self.logger_tc.info(f"POLICY_INTERFACE occupied with: {message}")
            self.policy_interface_commdown.put(command)
            self.logger_tc.info(f"Sent local command: {command}")



        else:
            self.logger_tc.warning(f"Unknown interface in command: {interface_name}")


    ###################################################################
    # 6) Connecting to Sub-Interfaces
    ###################################################################
    def connect_save_interface(self):
        if self.save_interface_process is not None:
            self.logger_tc.info("SAVE_INTERFACE is already running.")
            return

        self.save_interface_occupied = "initialization"

        self.logger_tc.info("Starting SAVE_INTERFACE process.")
        self.save_interface_process = multiprocessing.Process(
            target=self.run_save_interface,
            args=(self.save_interface_commup,
                  self.save_interface_commdown,
                  self.shm_joint_data1,
                  self.color_buffers1,
                  self.depth_buffers1,
                  self.setup_id)  # Pass setup_id for config isolation
        )
        self.save_interface_process.start()

        self.checking_status = True
        self.commup_thread_save_interface = threading.Thread(
            target=self._watch_save_interface_status, daemon=True
        )
        self.commup_thread_save_interface.start()

        # Wait for acceptance
        while self.save_interface_occupied is not False:
            time.sleep(self.status_refresh_period)

        # if no success, disconnect all connected interfaces
        if self.success == False:
            self.disconnect_interfaces()
            self.logger_tc.error("Failed to connect to SAVE_INTERFACE.")
            return
        self.success = False

        self.logger_tc.info("SAVE_INTERFACE is ready for commands.")

    def connect_teachbot_interface(self):
        # Return early if teachbot is not configured
        if not self.has_teachbot:
            self.logger_tc.info("Teachbot interface not configured, skipping connection.")
            return
            
        if self.teachbot_interface_process is not None:
            self.logger_tc.info("TEACHBOT_INTERFACE is already running.")
            return

        self.teachbot_interface_occupied = "initialization"

        self.logger_tc.info("Starting TEACHBOT_INTERFACE process.")
        self.teachbot_interface_process = multiprocessing.Process(
            target=self.run_teachbot_interface,
            args=(self.teachbot_interface_commup,
                  self.teachbot_interface_commdown,
                  self.shm_target_pos1,
                  self.setup_id)
        )
        self.teachbot_interface_process.start()

        self.checking_status = True
        self.commup_thread_teachbot_interface = threading.Thread(
            target=self._watch_teachbot_interface_status, daemon=True
        )
        self.commup_thread_teachbot_interface.start()

        # Wait for acceptance
        while self.teachbot_interface_occupied is not False:
            time.sleep(self.status_refresh_period)

        # if no success, disconnect all connected interfaces
        if self.success == False:
            self.disconnect_interfaces()
            self.logger_tc.error("Failed to connect to TEACHBOT_INTERFACE.")
            return
        self.success = False

        self.logger_tc.info("TEACHBOT_INTERFACE is ready for commands.")

    def connect_robot_interface(self):
        if self.robot_interface_process is not None:
            self.logger_tc.info("ROBOT_INTERFACE is already running.")
            return

        self.robot_interface_occupied = "initialization"

        self.logger_tc.info("Starting ROBOT_INTERFACE process.")
        
        # Get shm_target_pos2 shared memory info to pass to robot interface
        shm_target_pos2_info = self.get_shm_target_pos2_info()
        
        self.robot_interface_process = multiprocessing.Process(
            target=self.run_robot_interface,
            args=(self.robot_interface_commup,
                  self.robot_interface_commdown,
                  self.shm_target_pos1,
                  shm_target_pos2_info,  # Pass shared memory info instead of object
                  self.shm_joint_data1,
                  self.shm_joint_data2,
                  self.setup_id)  # Pass setup_id for config isolation
        )
        self.robot_interface_process.start()

        self.checking_status = True
        self.commup_thread_robot_interface = threading.Thread(
            target=self._watch_robot_interface_status, daemon=True
        )
        self.commup_thread_robot_interface.start()

        # Wait for acceptance
        while self.robot_interface_occupied is not False:
            time.sleep(self.status_refresh_period)

        # if no success, disconnect all connected interfaces
        if self.success == False:
            self.disconnect_interfaces()
            self.logger_tc.error("Failed to connect to ROBOT_INTERFACE.")
            return
        self.success = False

        self.logger_tc.info("ROBOT_INTERFACE is ready for commands.")

    def connect_camera_interface(self):
        """
        Launch a separate camera interface process for each configured camera.
        Each camera gets its own process, communication queues, and image queues.
        """
        if not self.camera_cfgs:
            self.logger_tc.info("No cameras configured, skipping camera interface setup.")
            return

        # Check if camera interfaces are already running
        if hasattr(self, 'camera_processes') and self.camera_processes:
            self.logger_tc.info("CAMERA_INTERFACE processes are already running.")
            return

        # Recreate ring buffers if they don't exist or were unlinked
        try:
            # Check if ring buffers still exist by trying to access them
            for camera in self.camera_cfgs:
                camera_name = camera["name"]
                if (not hasattr(self, 'color_buffers1') or 
                    camera_name not in self.color_buffers1 or 
                    not hasattr(self.color_buffers1[camera_name], 'shm')):
                    raise AttributeError("Ring buffers need to be recreated")
        except (AttributeError, FileNotFoundError):
            self.logger_tc.info("Recreating camera ring buffers for restart...")
            if hasattr(self, 'camera_ring_buffer_manager'):
                self.camera_ring_buffer_manager = CameraRingBufferManager()
            else:
                self.camera_ring_buffer_manager = CameraRingBufferManager()
            
            buffers_info = self.camera_ring_buffer_manager.create_buffers(
                self.camera_cfgs, 
                record_duration=self.record_duration,
                policy_img_buffer_size=self.policy_img_buffer_size
            )
            
            # Store ring buffer references
            self.color_buffers1 = buffers_info["color_buffers1"]
            self.depth_buffers1 = buffers_info["depth_buffers1"]
            self.color_buffers2 = buffers_info["color_buffers2"]
            self.depth_buffers2 = buffers_info["depth_buffers2"]
            
            self.logger_tc.info("Camera ring buffers recreated successfully.")
            # Small delay to ensure ring buffers are fully initialized
            time.sleep(0.1)

        self.camera_interface_occupied = "initialization"
        self.camera_processes = {}

        self.logger_tc.info("Starting CAMERA_INTERFACE processes for each camera.")

        for camera in self.camera_cfgs:
            camera_name = camera["name"]
            serial_number = camera["serial_number"]
            
            self.logger_tc.info(f"Starting camera interface process for '{camera_name}' (SN: {serial_number})")
            
            # Create a separate process for this camera
            camera_process = multiprocessing.Process(
                target=self.run_camera_interface,
                args=(self.camera_interface_commup,
                      self.camera_interface_commdown[camera_name],  # Individual command queue
                      self.color_buffers1[camera_name].name,   # Pass ring buffer name
                      self.depth_buffers1[camera_name].name,   # Pass ring buffer name
                      self.color_buffers2[camera_name].name,   # Pass policy image buffer name
                      self.depth_buffers2[camera_name].name,   # Pass policy depth buffer name
                      camera,  # Pass the camera config
                      self.setup_id)  # Pass setup_id for config isolation
            )
            camera_process.start()
            self.camera_processes[camera_name] = camera_process
            
            self.logger_tc.info(f"Camera process for '{camera_name}' started with PID: {camera_process.pid}")

        # Start a single status monitoring thread for all camera processes
        self.checking_status = True
        self.commup_thread_camera_interface = threading.Thread(
            target=self._watch_camera_interface_status, daemon=True
        )
        self.commup_thread_camera_interface.start()

        # Wait for all cameras to initialize
        while self.camera_interface_occupied is not False:
            time.sleep(self.status_refresh_period)

        # Check if initialization was successful
        if self.success == False:
            self.disconnect_interfaces()
            self.logger_tc.error("Failed to connect to CAMERA_INTERFACE.")
            return
        self.success = False

        self.logger_tc.info("All CAMERA_INTERFACE processes are ready for commands.")

    def _get_buffer_info_dict(self, buffer_dict):
            """
            Convert a dict of CameraRingBuffer objects to a dict of buffer info dicts.
            """
            return {
                name: {
                    "name": buf.name,
                    "width": buf.width,
                    "height": buf.height,
                    "channels": buf.channels,
                    "capacity": buf.capacity
                }
                for name, buf in buffer_dict.items()
            }

    def connect_policy_interface(self):
        """
        Start the policy interface process.
        """
        if not self.has_policy:
            self.logger_tc.info("Policy functionality disabled, skipping policy interface connection.")
            return

        self.logger_tc.info("Connecting to POLICY_INTERFACE...")

        if self.policy_interface_process is not None:
            self.logger_tc.info("POLICY_INTERFACE is already running.")
            return

        self.policy_interface_occupied = "initialization"

        shm_target_pos2_info = self.get_shm_target_pos2_info()
        shm_cpp_joint_data2_info = self.get_shm_cpp_joint_data2_info()
        

        self.logger_tc.info("Starting POLICY_INTERFACE process.")
        color_buffers2_info = self._get_buffer_info_dict(self.color_buffers2)
        depth_buffers2_info = self._get_buffer_info_dict(self.depth_buffers2)
        self.policy_interface_process = multiprocessing.Process(
            target=self.run_policy_interface,
            args=(self.policy_interface_commup,
                  self.policy_interface_commdown,
                  color_buffers2_info,  # Pass buffer info dict
                  depth_buffers2_info,  # Pass buffer info dict
                  shm_target_pos2_info,
                  self.shm_joint_data2,  # Python queue or None for C++ mode
                  shm_cpp_joint_data2_info,  # C++ shared memory info or None for Python mode
                  self.setup_id)  # Pass setup_id for config isolation
        )
        self.policy_interface_process.start()

        self.checking_status = True
        self.commup_thread_policy_interface = threading.Thread(
            target=self._watch_policy_interface_status, daemon=True
        )
        self.commup_thread_policy_interface.start()

        # Wait for acceptance
        while self.policy_interface_occupied is not False:
            time.sleep(self.status_refresh_period)

        # if no success, disconnect all connected interfaces
        if self.success == False:
            self.disconnect_interfaces()
            self.logger_tc.error("Failed to connect to POLICY_INTERFACE.")
            return
        self.success = False

        self.logger_tc.info("POLICY_INTERFACE is ready for commands.")


    ###################################################################
    # 7) Status Watchers
    ###################################################################
    def _watch_teachbot_interface_status(self):
        """
        Reads dictionary messages from teachbot_interface_commup.
        If we see "type=RESP", "interface=TEACHBOT_INTERFACE", 
        then check error. If there's an error, forward it to the UI
        and disconnect everything. Otherwise, if it matches the
        occupant message, clear occupant status.
        """
        self.logger_tc.info("Teachbot interface status thread started.")
        while self.checking_status:
            while not self.teachbot_interface_commup.empty():
                payload = self.teachbot_interface_commup.get()
                setup_info = f" (setup {payload.get('setup_id', 'unknown')})" if payload.get('setup_id') else ""
                self.logger_tc.info("Received message%s: %s", setup_info, payload)

                # Expecting dict with keys: type, interface, message, error, etc.
                msg_type = payload.get("type", "")
                interface = payload.get("interface", "")
                msg_message = payload.get("message", "")
                msg_error = payload.get("error", "")

                # Must match TEACHBOT_INTERFACE
                if msg_type == "RESP" and (interface == "TEACHBOT_INTERFACE" or interface.endswith("_TEACHBOT_INTERFACE")):
                    # Check for errors FIRST, before clearing occupied flag
                    if msg_error != "None":
                        self.success = False
                        self.teachbot_interface_occupied = False
                        self.logger_tc.error(f"TEACHBOT_INTERFACE error from '{msg_message}': {msg_error}")
                        self.send_response(payload, error=msg_error)
                    elif msg_message == self.teachbot_interface_occupied:
                        self.success = True
                        self.teachbot_interface_occupied = False
                        self.logger_tc.info("TEACHBOT_INTERFACE message: %s, completed. success = %s", msg_message, self.success)
                    else:
                        self.logger_tc.info("Untracked TEACHBOT_INTERFACE response: %s", msg_message)

                elif msg_type == "CMD" and (interface == "TEACHBOT_INTERFACE" or interface.endswith("_TEACHBOT_INTERFACE")):
                    # Example: If the robot signals a 'stop' command to be re-broadcast
                    if msg_message == "stop":
                        full_message = {"type": "CMD", "interface": self.get_interface_name("TEACHBOT_INTERFACE"), "message": "stop"}
                        self.publish_controller_command(full_message)

                else:
                    self.logger_tc.warning("Unknown message from TEACHBOT_INTERFACE: %s", payload)
            time.sleep(self.check_queue_period)

    def _watch_robot_interface_status(self):
        """
        Reads dictionary messages from robot_interface_commup.
        If we see "type=RESP", "interface=ROBOT_INTERFACE", 
        then check error. If there's an error, forward entire payload to UI
        and disconnect everything.
        """
        self.logger_tc.info("Robot interface status thread started.")
        while self.checking_status:
            while not self.robot_interface_commup.empty():
                payload = self.robot_interface_commup.get()
                setup_info = f" (setup {payload.get('setup_id', 'unknown')})" if payload.get('setup_id') else ""
                self.logger_tc.info("Received message from robot interface%s: %s", setup_info, payload)

                msg_type = payload.get("type", "")
                interface = payload.get("interface", "")
                msg_message = payload.get("message", "")
                msg_error = payload.get("error", "None")

                # Must match ROBOT_INTERFACE
                if msg_type == "RESP" and (interface == "ROBOT_INTERFACE" or interface.endswith("_ROBOT_INTERFACE")):
                    # Check for errors FIRST, before clearing occupied flag
                    if msg_error != "None":
                        self.success = False
                        self.robot_interface_occupied = False
                        self.logger_tc.error(f"ROBOT_INTERFACE error from '{msg_message}': {msg_error}")
                        self.send_response(payload, error=msg_error)
                    elif msg_message == self.robot_interface_occupied:
                        self.success = True
                        self.robot_interface_occupied = False
                        self.logger_tc.info("ROBOT_INTERFACE message: %s, completed. success = %s", msg_message, self.success)
                    else:
                        self.logger_tc.info("Untracked ROBOT_INTERFACE response: %s", msg_message)


                elif msg_type == "CMD" and (interface == "ROBOT_INTERFACE" or interface.endswith("_ROBOT_INTERFACE")):
                    # Example: If the robot signals a 'stop' command to be re-broadcast
                    if msg_message == "stop":
                        full_message = {"type": "CMD", "interface": self.get_interface_name("ROBOT_INTERFACE"), "message": "stop"}
                        self.publish_controller_command(full_message)
                else:
                    self.logger_tc.warning("Unknown message from ROBOT_INTERFACE: %s", payload)
            time.sleep(self.check_queue_period)

    def _watch_save_interface_status(self):
        """
        Reads dictionary messages from save_interface_commup.
        If we see "type=RESP", "interface=SAVE_INTERFACE",
        then check error. If there's an error, forward entire payload to UI
        and disconnect everything.
        """
        self.logger_tc.info("Save interface status thread started.")
        while self.checking_status:
            while not self.save_interface_commup.empty():
                payload = self.save_interface_commup.get()
                self.logger_tc.info("Received message from save interface: %s", payload)

                msg_type = payload.get("type", "")
                interface = payload.get("interface", "")
                msg_message = payload.get("message", "")
                msg_error = payload.get("error", "None")

                # Must match SAVE_INTERFACE (with or without setup prefix)
                if msg_type == "RESP" and (interface == "SAVE_INTERFACE" or interface.endswith("_SAVE_INTERFACE")):
                    # Check for errors FIRST, before clearing occupied flag
                    if msg_error != "None":
                        self.success = False
                        self.save_interface_occupied = False
                        self.logger_tc.error(f"SAVE_INTERFACE error from '{msg_message}': {msg_error}")
                        self.send_response(payload, error=msg_error)
                    elif msg_message == self.save_interface_occupied:
                        self.success = True
                        self.save_interface_occupied = False
                        self.logger_tc.info("SAVE_INTERFACE message: %s, completed. success = %s", msg_message, self.success)
                    else:
                        self.logger_tc.info("Untracked SAVE_INTERFACE response: %s", msg_message)

                elif msg_type == "CMD" and (interface == "SAVE_INTERFACE" or interface.endswith("_SAVE_INTERFACE")):
                    # Example: If the robot signals a 'stop' command to be re-broadcast
                    if msg_message == "stop":
                        full_message = {"type": "CMD", "interface": self.get_interface_name("SAVE_INTERFACE"), "message": "stop"}
                        self.publish_controller_command(full_message)

                else:
                    self.logger_tc.warning("Unknown message from SAVE_INTERFACE: %s", payload)
            time.sleep(self.check_queue_period)

    def _watch_camera_interface_status(self):
        """
        Reads dictionary messages from camera_interface_commup.
        Monitors status from all camera interface processes.
        If we see "type=RESP", "interface=CAMERA_INTERFACE", 
        then check error. If there's an error, forward entire payload to UI
        and disconnect everything.
        """
        self.logger_tc.info("Camera interface status thread started.")
        cameras_initialized = set()
        cameras_completed_current_cmd = set()
        total_cameras = len(self.camera_cfgs)
        
        while self.checking_status:
            while not self.camera_interface_commup.empty():
                payload = self.camera_interface_commup.get()
                self.logger_tc.info("Received message from camera interface: %s", payload)

                msg_type = payload.get("type", "")
                interface = payload.get("interface", "")
                msg_message = payload.get("message", "")
                msg_error = payload.get("error", "None")
                camera_name = payload.get("camera_name", "")

                # Must match CAMERA_INTERFACE
                if msg_type == "RESP" and (interface == "CAMERA_INTERFACE" or interface.endswith("_CAMERA_INTERFACE")):
                    # Check for errors FIRST, before any state changes
                    if msg_error != "None":
                        self.success = False
                        self.camera_interface_occupied = False
                        self.logger_tc.error(f"CAMERA_INTERFACE error from '{camera_name}': {msg_error}")
                        self.send_response(payload, error=msg_error)
                    # Only track initialization responses during initialization phase
                    elif msg_message == "initialization" and self.camera_interface_occupied == "initialization":
                        cameras_initialized.add(camera_name)
                        self.logger_tc.info(f"Camera '{camera_name}' initialized. ({len(cameras_initialized)}/{total_cameras})")
                        
                        # All cameras initialized
                        if len(cameras_initialized) >= total_cameras:
                            self.success = True
                            self.camera_interface_occupied = False
                            self.logger_tc.info("All camera interfaces initialized successfully.")
                            
                    elif msg_message == self.camera_interface_occupied and msg_message != "initialization":
                        # Handle non-initialization command responses
                        self.logger_tc.info(f"Camera '{camera_name}' completed: {msg_message}")
                        cameras_completed_current_cmd.add(camera_name)
                        
                        # Check if all cameras completed the current command
                        if len(cameras_completed_current_cmd) >= total_cameras:
                            self.logger_tc.info(f"All cameras completed command: {msg_message}")
                            self.camera_interface_occupied = False
                            cameras_completed_current_cmd.clear()  # Reset for next command
                            
                    else:
                        self.logger_tc.info("Untracked CAMERA_INTERFACE response: %s", msg_message)

                else:
                    self.logger_tc.warning("Unknown message from CAMERA_INTERFACE: %s", payload)
            time.sleep(self.check_queue_period)
        
    def _watch_policy_interface_status(self):
        """
        Reads dictionary messages from policy_interface_commup.
        If we see "type=RESP", "interface=POLICY_INTERFACE", 
        then check error. If there's an error, forward entire payload to UI
        and disconnect everything.
        """
        self.logger_tc.info("Policy interface status thread started.")
        while self.checking_status:
            while not self.policy_interface_commup.empty():
                payload = self.policy_interface_commup.get()
                self.logger_tc.info("Received message from policy interface: %s", payload)

                msg_type = payload.get("type", "")
                interface = payload.get("interface", "")
                msg_message = payload.get("message", "")
                msg_error = payload.get("error", "None")

                # Must match POLICY_INTERFACE
                if msg_type == "RESP" and (interface == "POLICY_INTERFACE" or interface.endswith("_POLICY_INTERFACE")):
                    # Check for errors FIRST, before clearing occupied flag
                    if msg_error != "None":
                        self.success = False
                        self.policy_interface_occupied = False
                        self.logger_tc.error(f"POLICY_INTERFACE error from '{msg_message}': {msg_error}")
                        self.send_response(payload, error=msg_error)
                    elif msg_message == self.policy_interface_occupied:
                        self.success = True
                        self.policy_interface_occupied = False
                        self.logger_tc.info("POLICY_INTERFACE message: %s, completed. success = %s", msg_message, self.success)
                    else:
                        self.logger_tc.info("Untracked POLICY_INTERFACE response: %s", msg_message)

                else:
                    self.logger_tc.warning("Unknown message from POLICY_INTERFACE: %s", payload)
            time.sleep(self.check_queue_period)



    ###################################################################
    # 8) Disconnect + Cleanup
    ###################################################################
    def disconnect_interfaces_only(self):
        """
        Disconnect interface processes but keep this controller running.
        """
        self.disconnect_interfaces()

        # Stop watchers
        self.checking_status = False
        self._join_status_threads()

        self.logger_tc.info("All interfaces disconnected; controller is still running.")

    def disconnect_all(self):
        """
        Fully disconnect everything and call cleanup => full shutdown.
        """
        self.disconnect_interfaces()
        self.cleanup()

    def disconnect_interfaces(self):
        self.logger_tc.info("Disconnecting all interfaces...")
        self._disconnect_interface(self.robot_interface_process, self.get_interface_name("ROBOT_INTERFACE"))
        self._disconnect_interface(self.policy_interface_process, self.get_interface_name("POLICY_INTERFACE"))
        self._disconnect_interface(self.teachbot_interface_process, self.get_interface_name("TEACHBOT_INTERFACE"))
        self._disconnect_interface(self.save_interface_process, self.get_interface_name("SAVE_INTERFACE"))
        self._disconnect_camera_interfaces()

        self.checking_status = False
    
    def _disconnect_interface(self, process_obj, interface_name):
        """
        Helper to send a disconnect command, wait, then shut down the process.
        """
        if process_obj is None:
            self.logger_tc.info(f"{interface_name} not running; nothing to disconnect.")
            return

        # For TEACHBOT_INTERFACE, the final command is "stop" per original logic
        disconnect_cmd = {
            "type": "CMD",
            "interface": interface_name,
            "message": "stop"
        }

        self.send_command(disconnect_cmd)
        self.logger_tc.info(f"Disconnecting {interface_name}...")

        # For TEACHBOT_INTERFACE, wait for its occupant flag to clear
        base_interface_name = self.get_base_interface_name(interface_name)
        if base_interface_name == "ROBOT_INTERFACE":
            while self.robot_interface_occupied is not False:
                time.sleep(self.status_refresh_period)
            self.robot_interface_process = None
        elif base_interface_name == "TEACHBOT_INTERFACE":
            while self.teachbot_interface_occupied is not False:
                time.sleep(self.status_refresh_period)
            self.teachbot_interface_process = None
        elif base_interface_name == "SAVE_INTERFACE":
            while self.save_interface_occupied is not False:
                time.sleep(self.status_refresh_period)
            self.save_interface_process = None
        elif base_interface_name == "POLICY_INTERFACE":
            while self.policy_interface_occupied is not False:
                time.sleep(self.status_refresh_period)
            self.policy_interface_process = None


        self.logger_tc.info(f"{interface_name} disconnected.")

    def _disconnect_camera_interfaces(self):
        """
        Disconnect all camera interface processes.
        """
        if not hasattr(self, 'camera_processes') or not self.camera_processes:
            self.logger_tc.info("No camera interfaces to disconnect.")
            return

        # Send stop command to all camera processes
        disconnect_cmd = {
            "type": "CMD",
            "interface": self.get_interface_name("CAMERA_INTERFACE"),
            "message": "stop"
        }
        
        self.send_command(disconnect_cmd)
        self.logger_tc.info("Disconnecting all camera interfaces...")

        # Wait for camera interface to stop
        while self.camera_interface_occupied is not False:
            time.sleep(self.status_refresh_period)

        # Terminate and join all camera processes (similar to other interface processes)
        for camera_name, process in self.camera_processes.items():
            if process and process.is_alive():
                try:
                    process.join(timeout=5.0)
                    if process.is_alive():
                        self.logger_tc.warning(f"Camera process for '{camera_name}' did not terminate gracefully, forcing termination.")
                        process.terminate()
                        process.join(timeout=2.0)
                        if process.is_alive():
                            process.kill()
                            process.join()
                    self.logger_tc.info(f"Camera process for '{camera_name}' terminated.")
                except Exception as e:
                    self.logger_tc.error(f"Error terminating camera process for '{camera_name}': {e}")
        
        # Clear camera processes
        self.camera_processes = {}
        
        self.logger_tc.info("All camera interfaces disconnected.")

    def _join_status_threads(self):
        """
        Joins the interface status-checking threads.
        """
        if self.commup_thread_robot_interface:
            self.commup_thread_robot_interface.join(timeout=5.0)
            self.logger_tc.info("ROBOT_INTERFACE status thread joined.")
            self.commup_thread_robot_interface = None

        if self.commup_thread_teachbot_interface:
            self.commup_thread_teachbot_interface.join(timeout=5.0)
            self.logger_tc.info("TEACHBOT_INTERFACE status thread joined.")
            self.commup_thread_teachbot_interface = None

        if self.commup_thread_save_interface:
            self.commup_thread_save_interface.join(timeout=5.0)
            self.logger_tc.info("SAVE_INTERFACE status thread joined.")
            self.commup_thread_save_interface = None

        if hasattr(self, 'commup_thread_camera_interface') and self.commup_thread_camera_interface:
            self.commup_thread_camera_interface.join(timeout=5.0)
            self.logger_tc.info("CAMERA_INTERFACE status thread joined.")
            self.commup_thread_camera_interface = None

        if self.commup_thread_policy_interface:
            self.commup_thread_policy_interface.join(timeout=5.0)
            self.logger_tc.info("POLICY_INTERFACE status thread joined.")
            self.commup_thread_policy_interface = None

    def cleanup(self):
        """
        Complete shutdown: stop threads, stop processes, set flags, etc.
        """
        self.logger_tc.info("Controller cleanup initiated.")
        self.checking_status = False

        self._join_status_threads()

        self._stop_interface_process(self.robot_interface_process, self.get_interface_name("ROBOT_INTERFACE"))
        self.robot_interface_process = None

        if self.has_teachbot:
            self._stop_interface_process(self.teachbot_interface_process, self.get_interface_name("TEACHBOT_INTERFACE"))
            self.teachbot_interface_process = None

        self._stop_interface_process(self.save_interface_process, self.get_interface_name("SAVE_INTERFACE"))
        self.save_interface_process = None

        # stop camera interfaces
        if hasattr(self, 'camera_processes') and self.camera_processes:
            for camera_name, process in self.camera_processes.items():
                self._stop_interface_process(process, self.get_interface_name(f"CAMERA_INTERFACE_{camera_name}"))
            self.camera_processes = {}

        # stop policy interface
        self._stop_interface_process(self.policy_interface_process, self.get_interface_name("POLICY_INTERFACE"))
        self.policy_interface_process = None

        # Clean up shared memory for C++ control loop
        if self.control_loop_language == "cpp":
            self.cleanup_shared_memory_cpp()

        # Clean up policy shared memory
        self.cleanup_shm_target_pos2()

        # Cleanup camera interfaces and shared memory
        self._disconnect_camera_interfaces()
        self.cleanup_shared_memory_cameras()


        self.processes_running = False
        self._join_main_threads()

        self.cleanup_done = True
        self.logger_tc.info("RobotController cleanup complete. Exiting.")

    def _stop_interface_process(self, process_obj, name):
        """
        Gracefully join the interface process if it's alive.
        """
        if process_obj is not None:
            process_obj.join(timeout=5.0)
            self.logger_tc.info(f"{name} process terminated.")

    def _join_main_threads(self):
        """
        Joins any remaining main consumer/publisher threads.
        """
        cur_thread = threading.current_thread()

        if self.status_pub_tc_thread and cur_thread != self.status_pub_tc_thread:
            self.status_pub_tc_thread.join(timeout=5.0)
            self.logger_tc.info("Status publisher thread joined.")
            self.status_pub_tc_thread = None

        if self.command_consumer_thread and cur_thread != self.command_consumer_thread:
            self.command_consumer_thread.join(timeout=5.0)
            self.logger_tc.info("Command consumer thread joined.")
            self.command_consumer_thread = None


    ####################################################################
    # 9) record mistake helpers
    ####################################################################

    def _process_mistake_files(self, dataset_name, original_payload):
        """
        Internal method that processes mistake files in a separate thread.
        """
        try:
            # Get configuration values
            episode_length = self.config["general"]["record_duration"]  # seconds
            saving_time = episode_length * 0.1  # 10% of episode length
            offset = 4  # standard 4 seconds offset
            
            # Calculate total threshold time
            recent_threshold = episode_length + saving_time + offset
            
            # Wait for episodes to complete (episode_length + saving_time)
            wait_time = episode_length + saving_time
            self.logger_tc.info(f"Waiting {wait_time} seconds for episodes to complete before moving files...")
            time.sleep(wait_time)
            
            # Get data directory
            parent_directory = get_data_path(self.config)
            
            # Process directories
            moved_files_count = self._move_files_to_mistakes(
                dataset_name, recent_threshold, parent_directory
            )
            
            # Send completion response to TOS UI
            response_payload = original_payload.copy()
            response_payload["type"] = "RESP"
            self.send_response(
                payload=response_payload,
                error="None", 
                status="completed",
                moved_files=moved_files_count,
                dataset_name=dataset_name
            )
            
            self.logger_tc.info(f"Mistake processing completed for dataset '{dataset_name}'. Moved {moved_files_count} files.")
            
        except Exception as e:
            error_msg = str(e)
            self.logger_tc.error(f"_process_mistake_files error: {error_msg}")
            
            # Send error response to TOS UI
            response_payload = original_payload.copy()
            response_payload["type"] = "RESP"
            self.send_response(payload=response_payload, error=error_msg)

    def _move_files_to_mistakes(self, dataset_name, recent_threshold, parent_directory):
        """
        Move files that are within the recent_threshold to '_mistakes' subdirectories.
        Returns the count of moved files.
        """
        current_time = time.time()
        moved_files_count = 0
        
        try:
            # Generate a list of directory paths that start with the dataset name
            if not os.path.exists(parent_directory):
                self.logger_tc.warning(f"Parent directory does not exist: {parent_directory}")
                return 0
                
            directories = [
                d for d in os.listdir(parent_directory) 
                if d.startswith(dataset_name) and os.path.isdir(os.path.join(parent_directory, d))
            ]
            
            # Process each relevant directory
            for dir_name in directories:
                if dir_name.endswith("_mistakes"):
                    continue  # Skip existing mistake directories
                    
                directory_path = os.path.join(parent_directory, dir_name)
                moved_count = self._move_directory_files_to_mistakes(
                    directory_path, recent_threshold, current_time
                )
                moved_files_count += moved_count
                
        except Exception as e:
            self.logger_tc.error(f"Error in _move_files_to_mistakes: {e}")
            raise
            
        return moved_files_count

    def _move_directory_files_to_mistakes(self, directory_path, recent_threshold, current_time):
        """
        Move files within a single directory that are within the recent_threshold to a '_mistakes' subdirectory.
        Returns the count of moved files.
        """
        moved_count = 0
        
        try:
            mistakes_directory = directory_path + '_mistakes'
            
            # Create the '_mistakes' directory if it does not exist
            if not os.path.exists(mistakes_directory):
                os.makedirs(mistakes_directory)
                self.logger_tc.info(f"Created mistakes directory: {mistakes_directory}")
            
            # Process each file in the directory
            if not os.path.exists(directory_path):
                return 0
                
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path):
                    creation_time = os.path.getctime(file_path)
                    if (current_time - creation_time) <= recent_threshold:
                        new_path = os.path.join(mistakes_directory, filename)
                        self.logger_tc.info(f"Moving file to mistakes directory: {filename}")
                        os.rename(file_path, new_path)
                        moved_count += 1
                        
        except Exception as e:
            self.logger_tc.error(f"Error moving files in directory {directory_path}: {e}")
            raise
            
        return moved_count
        





###################################################################
# 10) Helper Functions
###################################################################

@contextmanager
def open_channel(rabbit_conf, logger, client_name=None):
    """
    Context manager that yields a Pika channel, then cleans it up.
    """
    connection = robust_connect(rabbit_conf, logger, client_name=client_name)
    channel = connection.channel()
    try:
        yield channel
    finally:
        channel.close()
        connection.close()

def publish_message(rabbit_conf, logger, routing_key, message, client_name=None):
    """
    Ephemeral publish: opens a connection/channel, publishes, then closes.
    """
    with open_channel(rabbit_conf, logger, client_name) as channel:
        channel.basic_publish(
            exchange=rabbit_conf["exchange_name"],
            routing_key=routing_key,
            body=message.encode("utf-8") if isinstance(message, str) else message
        )

def robust_consume(rabbit_conf, logger, queue_name, routing_key, on_message_callback, stop_flag_func):
    """
    Repeatedly connect + channel.start_consuming().
    If disconnected, retry. If 'stop_flag_func()' is True, exit.
    """
    logger.info(f"Starting robust_consume on {queue_name} with key {routing_key}...")
    while not stop_flag_func():
        try:
            connection = robust_connect(rabbit_conf, logger, client_name=f"tc_consumer_{queue_name}")
            channel = connection.channel()

            channel.basic_consume(
                queue=queue_name,
                on_message_callback=on_message_callback,
                auto_ack=False
            )
            logger.info(f"Begin consuming commands on queue={queue_name}")
            channel.start_consuming()

        except (AMQPConnectionError, AMQPChannelError) as e:
            logger.warning(f"Consumer lost connection: {e}. Reconnecting in 1s...")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Unexpected error in robust_consume: {e}. Retrying in 1s...")
            time.sleep(1)
        finally:
            try: 
                channel.close()
            except:
                pass
            try:
                connection.close()
            except:
                pass

        if stop_flag_func():
            logger.info(f"Stop flag detected. Exiting consume loop for {queue_name}.")
            break

def robust_connect(rabbit_conf, logger, client_name=None):
    """
    Attempt to connect to RabbitMQ with advanced parameters:
      - client_properties => custom "connection_name"
      - heartbeat => keep the connection alive
      - blocked_connection_timeout => how long to wait if RabbitMQ is blocking
      - max_retries => 0 means infinite
      - wait_seconds => delay between retries
      - product / information => additional metadata
    """
    if not client_name:
        client_name = rabbit_conf["client_name_default"]
    attempts = 0
    creds = pika.PlainCredentials(rabbit_conf["user"], rabbit_conf["pass"])
    client_props = {
        "connection_name": client_name,
        "product": rabbit_conf["product"],
        "information": rabbit_conf["information"]
    }

    while True:
        try:
            params = pika.ConnectionParameters(
                host=rabbit_conf["host"],
                credentials=creds,
                heartbeat=rabbit_conf["heartbeat"],
                blocked_connection_timeout=rabbit_conf["blocked_connection_timeout"],
                client_properties=client_props
            )
            conn = pika.BlockingConnection(params)
            return conn
        except (AMQPConnectionError, AMQPChannelError) as e:
            attempts += 1
            logger.warning(
                f"[robust_connect] Connection failed: {e} (attempt {attempts}). "
                f"Waiting {rabbit_conf['wait_seconds']}s."
            )
            time.sleep(rabbit_conf["wait_seconds"])
            if rabbit_conf["max_retries"] > 0 and attempts >= rabbit_conf["max_retries"]:
                raise
