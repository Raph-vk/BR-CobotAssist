# my_app/modules/camera_inteld405.py

import time
import json
import threading
import signal
import numpy as np
import os
import sys
from multiprocessing import shared_memory
import pyrealsense2 as rs
from contextlib import contextmanager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.utils import setup_logging, load_config
from .cam_utils import CameraRingBuffer

@contextmanager
def timeout_context(seconds):
    """Context manager for timeout functionality."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def send_response(logger_ci, camera_interface_commup, payload, error="None"):
    """
    Helper function to send a response message to the controller.
    """
    response = {
        "type": "RESP",
        "interface": "CAMERA_INTERFACE", 
        "message": payload.get("message", ""),
        "camera_name": payload.get("camera_name", ""),
        "error": error
    }
    camera_interface_commup.put(response)
    logger_ci.info(f"Sent response: {response}")

def calculate_data_size(width, height, format, timestamp_size):
    """Calculate bytes per frame based on stream format."""
    if format == rs.format.rgb8 or format == rs.format.bgr8:
        return width * height * 3 + timestamp_size
    elif format == rs.format.z16:
        return width * height * 2 + timestamp_size
    else:
        raise ValueError(f"Unsupported format: {format}")


class CameraInterfaceIntelD405:
    """
    Interface for Intel RealSense D405 cameras.
    Manages camera initialization, frame capture, and communication with the controller.
    """

    def __init__(self, camera_interface_commup, camera_interface_commdown, 
                 color_buffer, depth_buffer, color_buffer2, depth_buffer2, 
                 camera_config, logger_ci, config):
        """
        :param camera_interface_commup: multiprocessing.Queue, from interface to controller
        :param camera_interface_commdown: multiprocessing.Queue, from controller to interface
        :param color_buffer: CameraRingBuffer for color images (recording)
        :param depth_buffer: CameraRingBuffer for depth images (recording)
        :param color_buffer2: CameraRingBuffer for color images (policy)
        :param depth_buffer2: CameraRingBuffer for depth images (policy)
        :param camera_config: dict with camera configuration
        :param logger_ci: logger instance
        :param config: global configuration
        """
        self.camera_interface_commup = camera_interface_commup
        self.camera_interface_commdown = camera_interface_commdown
        self.color_buffer = color_buffer
        self.depth_buffer = depth_buffer
        self.color_buffer2 = color_buffer2
        self.depth_buffer2 = depth_buffer2
        self.camera_config = camera_config
        self.logger_ci = logger_ci
        self.config = config
        
        # Camera properties
        self.camera_name = camera_config["name"]
        self.serial_number = camera_config["serial_number"]
        self.color_width = camera_config["color_width"]
        self.color_height = camera_config["color_height"]
        self.depth_width = camera_config["depth_width"]
        self.depth_height = camera_config["depth_height"]
        self.fps = camera_config["fps"]
        
        # RealSense pipeline
        self.pipeline = None
        self.config_rs = None
        self.device = None
        
        # Control flags
        self.running = False
        self.capturing = False
        self.capture_thread = None
        
        # Policy capture flags
        self.policy_capturing = False
        self.policy_capture_thread = None
        
        # Data sizes
        self.timestamp_size = np.dtype(np.float64).itemsize
        self.color_image_size = self.color_width * self.color_height * 3
        self.depth_image_size = self.depth_width * self.depth_height * 2
        
        self.logger_ci.info(f"CameraInterface initialized for '{self.camera_name}' (SN: {self.serial_number})")

        # Initialize camera
        if not self.initialize_camera():
            raise RuntimeError(f"Failed to initialize camera '{self.camera_name}'")
        self.running = True

    def hardware_reset(self):
        """
        Perform a hardware reset on the camera device.
        Returns True if successful, False otherwise.
        """
        try:
            self.logger_ci.info(f"Performing hardware reset on camera '{self.camera_name}'")
            
            # Find the device
            ctx = rs.context()
            devices = ctx.query_devices()
            device_found = False
            
            for device in devices:
                if device.get_info(rs.camera_info.serial_number) == self.serial_number:
                    device_found = True
                    device.hardware_reset()
                    self.logger_ci.info(f"Hardware reset completed for camera '{self.camera_name}'")
                    time.sleep(3)  # Wait for device to come back online
                    return True
                    
            if not device_found:
                self.logger_ci.error(f"Camera device with serial {self.serial_number} not found for reset")
                return False
                
        except Exception as e:
            self.logger_ci.error(f"Failed to perform hardware reset on camera '{self.camera_name}': {e}")
            return False

    ###################################################################
    # Camera interface commands
    ###################################################################

    def record_episodes(self):
        """
        Start recording episodes. This method can be extended to handle recording logic.
        """
        if not self.capturing:
            self.start_capture()
            self.logger_ci.info(f"Recording episodes started for camera '{self.camera_name}'")
        else:
            self.logger_ci.warning(f"Camera '{self.camera_name}' is already capturing")

    def record_episode(self):
        """
        Record a single episode. This method can be extended to handle episode recording logic.
        """
        if not self.capturing:
            self.start_capture()
            self.logger_ci.info(f"Recording single episode for camera '{self.camera_name}'")
        else:
            self.logger_ci.warning(f"Camera '{self.camera_name}' is already capturing")
            

    def stop(self):
        """
        Stop the camera interface and clean up resources.
        """
        self.logger_ci.info(f"Stopping camera interface for '{self.camera_name}'")
        
        # Stop both recording and policy capture
        if self.capturing:
            self.stop_capture()
        if self.policy_capturing:
            self.stop_policy_capture()
            
        self.cleanup()
        self.logger_ci.info(f"Camera interface for '{self.camera_name}' stopped successfully")
   
    def run_policy(self):
        """
        Start policy capture - fills color_buffer2 and depth_buffer2.
        """
        if not self.policy_capturing:
            self.start_policy_capture()
            self.logger_ci.info(f"Policy capture started for camera '{self.camera_name}'")
        else:
            self.logger_ci.warning(f"Camera '{self.camera_name}' is already doing policy capture")

    ####################################################################
    # Camera interface helper functions
    ####################################################################

    def initialize_camera(self, max_retries=3, retry_delay=5):
        """
        Initialize the RealSense camera pipeline with retry logic.
        
        Args:
            max_retries: Maximum number of initialization attempts
            retry_delay: Delay between retries in seconds
        """
        for attempt in range(max_retries):
            try:
                self.logger_ci.info(f"Initializing camera '{self.camera_name}' with serial {self.serial_number} (attempt {attempt + 1}/{max_retries})")
                
                # Check if camera is connected
                ctx = rs.context()
                devices = ctx.query_devices()
                device_found = False
                
                for device in devices:
                    if device.get_info(rs.camera_info.serial_number) == self.serial_number:
                        device_found = True
                        self.logger_ci.info(f"Found camera device: {device.get_info(rs.camera_info.name)}")
                        break
                
                if not device_found:
                    raise RuntimeError(f"Camera with serial number {self.serial_number} not found")
                
                # Initialize pipeline
                self.pipeline = rs.pipeline()
                self.config_rs = rs.config()
                self.config_rs.enable_device(self.serial_number)
                self.config_rs.enable_stream(rs.stream.color, self.color_width, self.color_height, 
                                           rs.format.rgb8, self.fps)
                self.config_rs.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, 
                                           rs.format.z16, self.fps)
                
                self.logger_ci.info(f"Starting pipeline for camera {self.camera_name}")
                pipeline_profile = self.pipeline.start(self.config_rs)
                self.device = pipeline_profile.get_device()
                
                # Enable global time if supported
                sensors = self.device.query_sensors()
                for sensor in sensors:
                    if sensor.supports(rs.option.global_time_enabled):
                        sensor.set_option(rs.option.global_time_enabled, 1)
                        self.logger_ci.info(f"Global time enabled for camera {self.camera_name}")

                # Wait for camera to stabilize
                time.sleep(5)
                
                self.logger_ci.info(f"Camera '{self.camera_name}' initialized successfully")
                return True
                
            except Exception as e:
                self.logger_ci.error(f"Failed to initialize camera '{self.camera_name}' on attempt {attempt + 1}: {e}")
                
                # Clean up any partially initialized resources
                try:
                    if hasattr(self, 'pipeline') and self.pipeline:
                        self.pipeline.stop()
                        self.pipeline = None
                except:
                    pass
                
                # If this is not the last attempt, perform hardware reset and retry
                if attempt < max_retries - 1:
                    self.logger_ci.info(f"Performing hardware reset before retry...")
                    if self.hardware_reset():
                        self.logger_ci.info(f"Hardware reset successful, retrying in {retry_delay} seconds...")
                    else:
                        self.logger_ci.warning(f"Hardware reset failed, retrying anyway in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger_ci.error(f"Failed to initialize camera '{self.camera_name}' after {max_retries} attempts")
                    return False
        
        return False

    def start_capture(self):
        """
        Start capturing frames in a separate thread.
        """
        if self.capturing:
            self.logger_ci.warning(f"Camera '{self.camera_name}' is already capturing")
            return
        
        self.capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.logger_ci.info(f"Started capture thread for camera '{self.camera_name}'")

    def stop_capture(self):
        """
        Stop capturing frames.
        """
        self.capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=5.0)
            self.capture_thread = None
        self.logger_ci.info(f"Stopped capture for camera '{self.camera_name}'")

    def _capture_loop(self):
        """
        Main capture loop that continuously reads frames and puts them in queues.
        """
        self.logger_ci.info(f"Starting capture loop for camera '{self.camera_name}'")
        
        
        while self.capturing and self.running:
            try:                
                # Step 1: Wait for frames
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                
                # Step 2: Get individual frames
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                                
                # Step 3: Convert frames to numpy arrays
                color_img = np.asanyarray(color_frame.get_data())
                depth_img = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
                
                # Step 4: Get timestamps
                color_timestamp = color_frame.get_timestamp() / 1000.0
                depth_timestamp = depth_frame.get_timestamp() / 1000.0
                
                # Write color frame to ring buffer
                color_success = self.color_buffer.write(
                    image=color_img,
                    timestamp=color_timestamp,
                    camera_name=self.camera_name,
                    serial_number=self.serial_number,
                    frame_type="color"
                )
                
                # Write depth frame to ring buffer  
                depth_success = self.depth_buffer.write(
                    image=depth_img,
                    timestamp=depth_timestamp,
                    camera_name=self.camera_name,
                    serial_number=self.serial_number,
                    frame_type="depth"
                )
                
                # With circular buffer behavior, writes should always succeed
                # Log warnings only if there are actual errors (not expected with new implementation)
                if not color_success:
                    self.logger_ci.error(f"Unexpected error writing to color ring buffer for camera '{self.camera_name}'")
                if not depth_success:
                    self.logger_ci.error(f"Unexpected error writing to depth ring buffer for camera '{self.camera_name}'")                

            except Exception as e:
                self.logger_ci.error(f"Error in capture loop cycle for camera '{self.camera_name}': {e}")
                time.sleep(0.1)  # Brief pause before retrying
        
    def start_policy_capture(self):
        """
        Start capturing frames for policy in a separate thread.
        """
        if self.policy_capturing:
            self.logger_ci.warning(f"Camera '{self.camera_name}' is already doing policy capture")
            return
        
        self.policy_capturing = True
        self.policy_capture_thread = threading.Thread(target=self._policy_capture_loop, daemon=True)
        self.policy_capture_thread.start()
        self.logger_ci.info(f"Started policy capture thread for camera '{self.camera_name}'")

    def stop_policy_capture(self):
        """
        Stop policy capturing frames.
        """
        self.policy_capturing = False
        if self.policy_capture_thread:
            self.policy_capture_thread.join(timeout=5.0)
            self.policy_capture_thread = None
        self.logger_ci.info(f"Stopped policy capture for camera '{self.camera_name}'")

    def _policy_capture_loop(self):
        """
        Main policy capture loop that continuously reads frames and puts them in policy buffers (buffer2).
        """
        self.logger_ci.info(f"Starting policy capture loop for camera '{self.camera_name}'")
        
        while self.policy_capturing and self.running:
            try:                
                # Step 1: Wait for frames
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                
                # Step 2: Get individual frames
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                # Step 3: Convert frames to numpy arrays
                color_img = np.asanyarray(color_frame.get_data())
                depth_img = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
                
                # Step 4: Get timestamps
                color_timestamp = color_frame.get_timestamp() / 1000.0
                depth_timestamp = depth_frame.get_timestamp() / 1000.0
                
                # Write color frame to policy ring buffer (buffer2)
                color_success = self.color_buffer2.write(
                    image=color_img,
                    timestamp=color_timestamp,
                    camera_name=self.camera_name,
                    serial_number=self.serial_number,
                    frame_type="color"
                )
                
                # Write depth frame to policy ring buffer (buffer2)
                depth_success = self.depth_buffer2.write(
                    image=depth_img,
                    timestamp=depth_timestamp,
                    camera_name=self.camera_name,
                    serial_number=self.serial_number,
                    frame_type="depth"
                )
                
                # With circular buffer behavior, writes should always succeed
                if not color_success:
                    self.logger_ci.error(f"Unexpected error writing to policy color ring buffer for camera '{self.camera_name}'")
                if not depth_success:
                    self.logger_ci.error(f"Unexpected error writing to policy depth ring buffer for camera '{self.camera_name}'")                

            except Exception as e:
                self.logger_ci.error(f"Error in policy capture loop cycle for camera '{self.camera_name}': {e}")
                time.sleep(0.1)  # Brief pause before retrying
        
    def cleanup(self):
        """
        Clean up camera resources.
        """
        self.logger_ci.info(f"Cleaning up camera '{self.camera_name}'")
        
        # First stop both capture threads and wait for them to complete
        self.stop_capture()
        self.stop_policy_capture()
        
        # Set running to False to signal cleanup
        self.running = False
        
        # Stop the camera pipeline
        if self.pipeline:
            try:
                self.pipeline.stop()
                self.logger_ci.info(f"Pipeline stopped for camera '{self.camera_name}'")
            except Exception as e:
                self.logger_ci.error(f"Error stopping pipeline for camera '{self.camera_name}': {e}")

        # Clean up ring buffer shared memory references
        try:
            if hasattr(self, 'color_buffer') and self.color_buffer:
                self.color_buffer.close(unlink=False)
                self.logger_ci.info(f"Color buffer closed for camera '{self.camera_name}'")
        except Exception as e:
            self.logger_ci.error(f"Error closing color buffer for camera '{self.camera_name}': {e}")
            
        try:
            if hasattr(self, 'depth_buffer') and self.depth_buffer:
                self.depth_buffer.close(unlink=False)
                self.logger_ci.info(f"Depth buffer closed for camera '{self.camera_name}'")
        except Exception as e:
            self.logger_ci.error(f"Error closing depth buffer for camera '{self.camera_name}': {e}")

        try:
            if hasattr(self, 'color_buffer2') and self.color_buffer2:
                self.color_buffer2.close(unlink=False)
                self.logger_ci.info(f"Color buffer2 closed for camera '{self.camera_name}'")
        except Exception as e:
            self.logger_ci.error(f"Error closing color buffer2 for camera '{self.camera_name}': {e}")
            
        try:
            if hasattr(self, 'depth_buffer2') and self.depth_buffer2:
                self.depth_buffer2.close(unlink=False)
                self.logger_ci.info(f"Depth buffer2 closed for camera '{self.camera_name}'")
        except Exception as e:
            self.logger_ci.error(f"Error closing depth buffer2 for camera '{self.camera_name}': {e}")
        
        self.logger_ci.info(f"Camera '{self.camera_name}' cleanup completed")


def run_camera_interface(camera_interface_commup, camera_interface_commdown, 
                        color_buffer_name, depth_buffer_name, 
                        color_buffer2_name, depth_buffer2_name, camera_config):
    """
    Main function to run a single camera interface process.
    
    :param camera_interface_commup: Queue for sending messages to controller
    :param camera_interface_commdown: Queue for receiving messages from controller
    :param color_buffer_name: Name of shared memory for color ring buffer (recording)
    :param depth_buffer_name: Name of shared memory for depth ring buffer (recording)
    :param color_buffer2_name: Name of shared memory for color ring buffer (policy)
    :param depth_buffer2_name: Name of shared memory for depth ring buffer (policy)
    :param camera_config: Dictionary with camera configuration
    """
    component_tag = "CAMERA_INTERFACE"
    camera_name = camera_config["name"]
    
    # Setup logging
    logger_ci = setup_logging(f"{component_tag}_{camera_name}")
    logger_ci.info(f"Starting Camera Interface for '{camera_name}'...")
    
    # Load configuration
    config = load_config()
    
    # Connect to existing ring buffers
    color_buffer = None
    depth_buffer = None
    camera_interface = None
    
    try:
        color_buffer = CameraRingBuffer(
            name=color_buffer_name,
            width=camera_config["color_width"],
            height=camera_config["color_height"],
            channels=3,
            capacity=0,  # Will be read from existing buffer
            create=False
        )
        
        depth_buffer = CameraRingBuffer(
            name=depth_buffer_name,
            width=camera_config["depth_width"],
            height=camera_config["depth_height"],
            channels=1,
            capacity=0,  # Will be read from existing buffer
            create=False
        )
        
        color_buffer2 = CameraRingBuffer(
            name=color_buffer2_name,
            width=camera_config["color_width"],
            height=camera_config["color_height"],
            channels=3,
            capacity=0,  # Will be read from existing buffer
            create=False
        )
        
        depth_buffer2 = CameraRingBuffer(
            name=depth_buffer2_name,
            width=camera_config["depth_width"],
            height=camera_config["depth_height"],
            channels=1,
            capacity=0,  # Will be read from existing buffer
            create=False
        )
        
        logger_ci.info(f"Connected to ring buffers: {color_buffer_name}, {depth_buffer_name}, {color_buffer2_name}, {depth_buffer2_name}")
        
    except Exception as e:
        logger_ci.error(f"Failed to connect to ring buffers: {e}")
        return
    
    # Instantiate camera interface
    try:
        camera_interface = CameraInterfaceIntelD405(
            camera_interface_commup, camera_interface_commdown,
            color_buffer, depth_buffer, color_buffer2, depth_buffer2, 
            camera_config, logger_ci, config
        )
        
        queue_check_period = config["general"]["check_queue_period"]
        
        # Notify controller that we're initialized and ready
        send_response(logger_ci, camera_interface_commup,
                     {"interface": component_tag, "message": "initialization", "camera_name": camera_name}, 
                     error="None")
        
    except Exception as e:
        err = f"Failed to initialize CameraInterface for '{camera_name}': {e}"
        logger_ci.error(err)
        send_response(logger_ci, camera_interface_commup,
                     {"interface": component_tag, "message": "initialization", "camera_name": camera_name}, 
                     error=err)
        return

    # Main command processing loop
    while camera_interface.running:
        if not camera_interface_commdown.empty():
            full_message = camera_interface_commdown.get()
            logger_ci.info("Received message: %s", full_message)

            msg_type = full_message.get("type", "")
            msg_interface = full_message.get("interface", "")
            message = full_message.get("message", "")

            if msg_type == "CMD" and msg_interface == "CAMERA_INTERFACE":
                try:
                    if message == "record_episodes":
                        camera_interface.record_episodes()
                        send_response(logger_ci, camera_interface_commup,
                                    {"interface": component_tag, "message": "record_episodes", "camera_name": camera_name},
                                    error="None")
                    
                    elif message == "record_episode":
                        camera_interface.record_episode()
                        send_response(logger_ci, camera_interface_commup,
                                    {"interface": component_tag, "message": "record_episode", "camera_name": camera_name},
                                    error="None")
                        
                    elif message == "run_policy":
                        camera_interface.run_policy()
                        send_response(logger_ci, camera_interface_commup,
                                    {"interface": component_tag, "message": "run_policy", "camera_name": camera_name},
                                    error="None")
                    
                    elif message == "stop":
                        logger_ci.info(f"Received stop command for camera '{camera_name}'")
                        camera_interface.stop()
                        send_response(logger_ci, camera_interface_commup,
                                    {"interface": component_tag, "message": "stop", "camera_name": camera_name},
                                    error="None")
                        break
                    
                    else:
                        logger_ci.warning(f"Unknown command: {message}")
                        send_response(logger_ci, camera_interface_commup,
                                    {"interface": component_tag, "message": message, "camera_name": camera_name},
                                    error=f"Unknown command: {message}")

                except Exception as e:
                    err = f"Error processing command '{message}' for camera '{camera_name}': {e}"
                    logger_ci.error(err)
                    send_response(logger_ci, camera_interface_commup,
                                {"interface": component_tag, "message": message, "camera_name": camera_name},
                                error=err)
            else:
                logger_ci.warning(f"Ignoring message not meant for CAMERA_INTERFACE: {full_message}")

        time.sleep(queue_check_period)

    logger_ci.info(f"Camera Interface for '{camera_name}' shutting down...")
