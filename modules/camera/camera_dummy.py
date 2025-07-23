# my_app/modules/camera_dummy.py

import time
import json
import threading
import signal
import numpy as np
import os
import sys
from multiprocessing import shared_memory
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
    if format == "rgb8" or format == "bgr8":
        return width * height * 3 + timestamp_size
    elif format == "z16":
        return width * height * 2 + timestamp_size
    else:
        raise ValueError(f"Unsupported format: {format}")


class CameraInterfaceDummy:
    """
    Dummy Interface for generating random noise images as camera data.
    Simulates 2 cameras generating random color and depth images.
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
        self.serial_number = camera_config.get("serial_number", f"DUMMY_{camera_config['name']}")
        self.color_width = camera_config["color_width"]
        self.color_height = camera_config["color_height"]
        self.depth_width = camera_config["depth_width"]
        self.depth_height = camera_config["depth_height"]
        self.fps = camera_config["fps"]
        
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
        
        # Frame timing
        self.frame_interval = 1.0 / self.fps
        
        self.logger_ci.info(f"Dummy CameraInterface initialized for '{self.camera_name}' (SN: {self.serial_number})")

        # Initialize dummy camera
        if not self.initialize_camera():
            raise RuntimeError(f"Failed to initialize dummy camera '{self.camera_name}'")
        self.running = True

    def hardware_reset(self):
        """
        Simulate a hardware reset on the dummy camera device.
        Returns True if successful, False otherwise.
        """
        try:
            self.logger_ci.info(f"Performing dummy hardware reset on camera '{self.camera_name}'")
            time.sleep(1)  # Simulate reset time
            self.logger_ci.info(f"Dummy hardware reset completed for camera '{self.camera_name}'")
            return True
                
        except Exception as e:
            self.logger_ci.error(f"Failed to perform dummy hardware reset on camera '{self.camera_name}': {e}")
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
            self.logger_ci.info(f"Recording episodes started for dummy camera '{self.camera_name}'")
        else:
            self.logger_ci.warning(f"Dummy camera '{self.camera_name}' is already capturing")

    def stop(self):
        """
        Stop the dummy camera interface and clean up resources.
        """
        self.logger_ci.info(f"Stopping dummy camera interface for '{self.camera_name}'")
        
        # Stop both recording and policy capture
        if self.capturing:
            self.stop_capture()
        if self.policy_capturing:
            self.stop_policy_capture()
            
        self.cleanup()
        self.logger_ci.info(f"Dummy camera interface for '{self.camera_name}' stopped successfully")
   
    def run_policy(self):
        """
        Start policy capture - fills color_buffer2 and depth_buffer2.
        """
        if not self.policy_capturing:
            self.start_policy_capture()
            self.logger_ci.info(f"Policy capture started for dummy camera '{self.camera_name}'")
        else:
            self.logger_ci.warning(f"Dummy camera '{self.camera_name}' is already doing policy capture")

    ####################################################################
    # Camera interface helper functions
    ####################################################################

    def initialize_camera(self, max_retries=3, retry_delay=2):
        """
        Initialize the dummy camera (just simulates initialization).
        
        Args:
            max_retries (int): Maximum number of initialization attempts
            retry_delay (int): Delay between retry attempts in seconds
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                self.logger_ci.info(f"Attempting to initialize dummy camera '{self.camera_name}' (attempt {attempt + 1}/{max_retries})")
                
                # Simulate initialization time
                time.sleep(0.5)
                
                self.logger_ci.info(f"Dummy camera '{self.camera_name}' initialized successfully")
                return True
                
            except Exception as e:
                self.logger_ci.error(f"Failed to initialize dummy camera '{self.camera_name}': {e}")
                
                # If this is not the last attempt, retry
                if attempt < max_retries - 1:
                    self.logger_ci.info(f"Retrying dummy camera initialization in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger_ci.error(f"Failed to initialize dummy camera '{self.camera_name}' after {max_retries} attempts")
                    return False
        
        return False

    def start_capture(self):
        """
        Start capturing frames in a separate thread.
        """
        if self.capturing:
            self.logger_ci.warning(f"Dummy camera '{self.camera_name}' is already capturing")
            return
        
        self.capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.logger_ci.info(f"Started capture thread for dummy camera '{self.camera_name}'")

    def stop_capture(self):
        """
        Stop capturing frames.
        """
        self.capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=5.0)
            self.capture_thread = None
        self.logger_ci.info(f"Stopped capture for dummy camera '{self.camera_name}'")

    def _generate_random_color_image(self):
        """Generate a random color image."""
        return np.random.randint(0, 256, 
                                (self.color_height, self.color_width, 3), 
                                dtype=np.uint8)

    def _generate_random_depth_image(self):
        """Generate a random depth image."""
        return np.random.randint(500, 5000, 
                                (self.depth_height, self.depth_width), 
                                dtype=np.uint16)

    def _capture_loop(self):
        """
        Main capture loop that continuously generates random frames and puts them in queues.
        """
        self.logger_ci.info(f"Starting capture loop for dummy camera '{self.camera_name}'")
        
        last_frame_time = time.time()
        
        while self.capturing and self.running:
            try:                
                # Control frame rate
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
                
                # Generate random frames
                color_img = self._generate_random_color_image()
                depth_img = self._generate_random_depth_image()
                
                # Generate timestamps
                timestamp = time.time()
                
                # Write color frame to ring buffer
                color_success = self.color_buffer.write(
                    image=color_img,
                    timestamp=timestamp,
                    camera_name=self.camera_name,
                    serial_number=self.serial_number,
                    frame_type="color"
                )
                
                # Write depth frame to ring buffer  
                depth_success = self.depth_buffer.write(
                    image=depth_img,
                    timestamp=timestamp,
                    camera_name=self.camera_name,
                    serial_number=self.serial_number,
                    frame_type="depth"
                )
                
                # Log warnings only if there are actual errors
                if not color_success:
                    self.logger_ci.error(f"Unexpected error writing to color ring buffer for dummy camera '{self.camera_name}'")
                if not depth_success:
                    self.logger_ci.error(f"Unexpected error writing to depth ring buffer for dummy camera '{self.camera_name}'")                

                last_frame_time = time.time()

            except Exception as e:
                self.logger_ci.error(f"Error in capture loop cycle for dummy camera '{self.camera_name}': {e}")
                time.sleep(0.1)  # Brief pause before retrying
        
    def start_policy_capture(self):
        """
        Start capturing frames for policy in a separate thread.
        """
        if self.policy_capturing:
            self.logger_ci.warning(f"Dummy camera '{self.camera_name}' is already doing policy capture")
            return
        
        self.policy_capturing = True
        self.policy_capture_thread = threading.Thread(target=self._policy_capture_loop, daemon=True)
        self.policy_capture_thread.start()
        self.logger_ci.info(f"Started policy capture thread for dummy camera '{self.camera_name}'")

    def stop_policy_capture(self):
        """
        Stop policy capturing frames.
        """
        self.policy_capturing = False
        if self.policy_capture_thread:
            self.policy_capture_thread.join(timeout=5.0)
            self.policy_capture_thread = None
        self.logger_ci.info(f"Stopped policy capture for dummy camera '{self.camera_name}'")

    def _policy_capture_loop(self):
        """
        Main policy capture loop that continuously generates random frames and puts them in policy buffers (buffer2).
        """
        self.logger_ci.info(f"Starting policy capture loop for dummy camera '{self.camera_name}'")
        
        last_frame_time = time.time()
        
        while self.policy_capturing and self.running:
            try:                
                # Control frame rate
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
                
                # Generate random frames
                color_img = self._generate_random_color_image()
                depth_img = self._generate_random_depth_image()
                
                # Generate timestamps
                timestamp = time.time()
                
                # Write color frame to policy ring buffer (buffer2)
                color_success = self.color_buffer2.write(
                    image=color_img,
                    timestamp=timestamp,
                    camera_name=self.camera_name,
                    serial_number=self.serial_number,
                    frame_type="color"
                )
                
                # Write depth frame to policy ring buffer (buffer2)
                depth_success = self.depth_buffer2.write(
                    image=depth_img,
                    timestamp=timestamp,
                    camera_name=self.camera_name,
                    serial_number=self.serial_number,
                    frame_type="depth"
                )
                
                # Log warnings only if there are actual errors
                if not color_success:
                    self.logger_ci.error(f"Unexpected error writing to policy color ring buffer for dummy camera '{self.camera_name}'")
                if not depth_success:
                    self.logger_ci.error(f"Unexpected error writing to policy depth ring buffer for dummy camera '{self.camera_name}'")                

                last_frame_time = time.time()

            except Exception as e:
                self.logger_ci.error(f"Error in policy capture loop cycle for dummy camera '{self.camera_name}': {e}")
                time.sleep(0.1)  # Brief pause before retrying
        
    def cleanup(self):
        """
        Clean up dummy camera resources.
        """
        self.logger_ci.info(f"Cleaning up dummy camera '{self.camera_name}'")
        
        # First stop both capture threads and wait for them to complete
        self.stop_capture()
        self.stop_policy_capture()
        
        # Set running to False to signal cleanup
        self.running = False
        
        # Clean up ring buffer shared memory references
        try:
            if hasattr(self, 'color_buffer') and self.color_buffer:
                self.color_buffer.close(unlink=False)
                self.logger_ci.info(f"Color buffer closed for dummy camera '{self.camera_name}'")
        except Exception as e:
            self.logger_ci.error(f"Error closing color buffer for dummy camera '{self.camera_name}': {e}")
            
        try:
            if hasattr(self, 'depth_buffer') and self.depth_buffer:
                self.depth_buffer.close(unlink=False)
                self.logger_ci.info(f"Depth buffer closed for dummy camera '{self.camera_name}'")
        except Exception as e:
            self.logger_ci.error(f"Error closing depth buffer for dummy camera '{self.camera_name}': {e}")

        try:
            if hasattr(self, 'color_buffer2') and self.color_buffer2:
                self.color_buffer2.close(unlink=False)
                self.logger_ci.info(f"Color buffer2 closed for dummy camera '{self.camera_name}'")
        except Exception as e:
            self.logger_ci.error(f"Error closing color buffer2 for dummy camera '{self.camera_name}': {e}")
            
        try:
            if hasattr(self, 'depth_buffer2') and self.depth_buffer2:
                self.depth_buffer2.close(unlink=False)
                self.logger_ci.info(f"Depth buffer2 closed for dummy camera '{self.camera_name}'")
        except Exception as e:
            self.logger_ci.error(f"Error closing depth buffer2 for dummy camera '{self.camera_name}': {e}")
        
        self.logger_ci.info(f"Dummy camera '{self.camera_name}' cleanup completed")

def run_camera_interface(camera_interface_commup, camera_interface_commdown, 
                        color_buffer_name, depth_buffer_name, 
                        color_buffer2_name, depth_buffer2_name, camera_config, setup_id=None):
    """
    Main function to run a single camera interface process.
    
    :param camera_interface_commup: Queue for sending messages to controller
    :param camera_interface_commdown: Queue for receiving messages from controller
    :param color_buffer_name: Name of shared memory for color ring buffer (recording)
    :param depth_buffer_name: Name of shared memory for depth ring buffer (recording)
    :param color_buffer2_name: Name of shared memory for color ring buffer (policy)
    :param depth_buffer2_name: Name of shared memory for depth ring buffer (policy)
    :param camera_config: Dictionary with camera configuration
    :param setup_id: Setup ID for config isolation
    """
    component_tag = "CAMERA_INTERFACE"
    camera_name = camera_config["name"]
    
    # Use setup-specific logging if setup_id is provided
    if setup_id is not None:
        component_tag = f"{component_tag}_{setup_id}"
    
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
        camera_interface = CameraInterfaceDummy(
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
