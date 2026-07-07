###################################################################
# Imports and Setup
###################################################################
import time
import numpy as np
import threading
from threading import Thread
import socket
import struct
from collections import deque
import json
import os
import sys
import re
import copy
import multiprocessing.shared_memory as shared_memory
import struct

# MQTT imports for external EE communication
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("Warning: paho-mqtt not available. External EE communication will be disabled.")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from utils.utils import setup_logging, load_config, get_data_path


###################################################################
# Helper Functions (Global)
###################################################################
def send_tc_command(queue, command):
    """
    Helper function to put a command into a queue.
    """
    queue.put(command)


def build_tmsct_packet(script_id, script_str):
    """
    Build a TMSCT packet in the format:
        $TMSCT,<len>,<ID>,<script>,*<CS>\r\n
    <CS> is a simple XOR checksum of all bytes from after '$' to before '*'.
    """
    data = f"{script_id},{script_str}"
    length_str = str(len(data))  # decimal length
    packet_body = f"TMSCT,{length_str},{data},"
    
    full_no_cs = f"${packet_body}*"

    # Compute XOR from after '$' to before '*'
    xor_val = 0
    for b in full_no_cs[1:-1].encode('ascii'):
        xor_val ^= b
    
    cs_str = f"{xor_val:02X}"
    packet = f"{full_no_cs}{cs_str}\r\n"
    return packet


def send_tmsct_cmd(sock, script_id, script_str, logger):
    """
    Send a TMSCT command via socket and optionally read back the robot's response.
    """
    packet_out = build_tmsct_packet(script_id, script_str)
    try:
        sock.sendall(packet_out.encode('ascii'))
        return True
    except Exception as e:
        logger.error(f"Error sending command: {e}")
        return False


def action_master_TM_translation(master_pos_deg, logger=None):
    """
    Transform master position to Techman robot coordinates.
    This function applies the necessary transformations for the Techman robot.
    """
    action = np.array(master_pos_deg, dtype=float)
    original = action.copy()
    
    action[2] = -action[2]
    action[4] = -action[4]
    action[2] = action[2] + 90
    action[3] = action[3] - 90
    action[4] = action[4] + 90
    action[5] = -action[5]
    
    return action


def action_master_safety_check(action_deg, upper_limits, lower_limits):
    """
    Safety check for joint positions within limits.
    """
    for i, (pos, upper, lower) in enumerate(zip(action_deg, upper_limits, lower_limits)):
        if pos < lower or pos > upper:
            return False
    return True


def round_position(position):
    """
    Round all values in a position list to 6 decimals.
    """
    return [round(pos, 6) for pos in position]


def send_response(logger_ti, robot_interface_commup, payload, error="None", **kwargs):
    """
    Helper to construct and send a consistent response dict:
        {
         "type": "RESP",
         "message": <whatever command was processed or context>,
         "error": <error string or "None">,
         ... plus optional extra fields ...
        }
    Then puts that response into the robot_interface_commup queue.
    """
    response = payload.copy() if isinstance(payload, dict) else {"message": str(payload)}
    response["type"] = "RESP"
    if error not in ("None", ""):
        response["error"] = error
    elif response.get("error", "") == "":
        response["error"] = "None"

    # Merge any additional kwargs
    response.update(kwargs)

    # Log and publish
    logger_ti.info(f"Preparing to send response: {response}")
    robot_interface_commup.put(response)
    logger_ti.info(f"Sent response: {response}")


##################################################################
# CLASS DEFINITION: TechmanRobot
##################################################################
class TechmanRobot():
    """
    Handles:
      - TMSCT connection (TCP) to a Techman robot,
      - UDP listening for status/feedback from robot,
      - Methods for motion commands (teleoperation, recording, playback),
      - Safety checks and limit monitoring,
      - Joint speed limiting based on config velocity_limits and safety factor,
      - Consistent structure matching FanucRobot implementation.
    """


    def __init__(self, robot_interface_commup, shm_target_pos1, shm_target_pos2_info, shm_joint_data1, shm_joint_data2, logger_ri, config):
        # Logging must be set first
        self.logger_ri = logger_ri

        # Component tag for interface identification
        self.component_tag = "ROBOT_INTERFACE"

        # Shared memory queues
        self.robot_interface_commup = robot_interface_commup
        self.shm_target_pos1 = shm_target_pos1
        self.shm_joint_data1 = shm_joint_data1
        self.shm_joint_data2 = shm_joint_data2

        # Attach to shm_target_pos2 shared memory segment
        self.shm_target_pos2_info = shm_target_pos2_info
        self.shm_target_pos2 = None
        if shm_target_pos2_info:
            try:
                self.shm_target_pos2 = shared_memory.SharedMemory(name=shm_target_pos2_info['name'])
                self.shm_target_pos2_capacity = shm_target_pos2_info['capacity']
                self.shm_target_pos2_entry_format = shm_target_pos2_info['entry_format']
                self.shm_target_pos2_entry_size = shm_target_pos2_info['entry_size']
                self.logger_ri.info(f"Attached to shm_target_pos2: {shm_target_pos2_info['name']}")
            except Exception as e:
                self.logger_ri.error(f"Failed to attach to shm_target_pos2: {e}")
                self.shm_target_pos2 = None

        # Config
        self.config = config

        # Config elements - using similar structure to FanucRobot
        self.status_refresh_period = config["general"]["status_refresh_period"]
        self.check_queue_period = config["general"]["check_queue_period"]
        self.control_dt = config["hardware"]["robot"]["control_dt"]
        self.check_queue_period_divisor = config["general"]["check_queue_period_divisor"]
        self.action_buffer_length = config["general"]["action_buffer_length"]
        self.start_position = config["general"]["start_position"]
        self.start_joint_tolerance = config["general"]["start_joint_tolerance"]
        self.dof = config["hardware"]["robot"]["dof"]
        self.dof_ee = config["hardware"]["robot"]["dof_ee"]
        self.total_dof = self.dof + self.dof_ee
        self.default_recording_speed = config["general"]["default_recording_speed"]
        self.gripper_treshold = config["general"]["gripper_treshold"]
        self.gripper_delay = config["general"]["gripper_delay"]
        self.action_buffer_length = config["general"]["action_buffer_length"]
        self.run_policy_active = False  # Initialize policy state
        
        # End Effector configuration
        self.ee_type = config["hardware"]["robot"].get("EE_type", "integrated")
        self.trigger_function = config["hardware"]["robot"].get("trigger_function", "position")
        
        # Techman-specific config
        self.robot_address = tuple(config["hardware"]["robot"]["robot_adress"])
        self.robot_recv_address = tuple(config["hardware"]["robot"]["robot_recv_address"])
        
        # Use default limits if not specified in config
        self.upper_limits = config["hardware"]["robot"].get("upper_limits", [180, 180, 180, 180, 180, 180])
        self.lower_limits = config["hardware"]["robot"].get("lower_limits", [-180, -180, -180, -180, -180, -180])
        
        # Joint velocity limits and safety factor
        self.velocity_limits = config["hardware"]["robot"].get("velocity_limits", [120, 120, 180, 180, 180, 180])
        self.limit_safety_factor = config["hardware"]["robot"].get("limit_safety_factor", 0.9)
        
        # Calculate safe velocity limits with safety factor applied
        self.safe_velocity_limits = [limit * self.limit_safety_factor for limit in self.velocity_limits]
        
       
        # Techman TMSCT Position command parameters
        move_params = config["hardware"]["robot"].get("move_to_start_params", {})
        self.move_accel_time = move_params.get("accel_time", 2000)
        self.move_motion_gain = move_params.get("motion_gain", 1)
        self.move_protection_time = move_params.get("protection_time", 20)
        
        continuous_params = config["hardware"]["robot"].get("continuous_mode_params", {})
        self.continuous_accel_time = continuous_params.get("accel_time", 2000)
        self.continuous_motion_gain = continuous_params.get("motion_gain", 1)
        self.continuous_protection_time = continuous_params.get("protection_time", 20)

        # State flags
        self.messages_sent = 0
        self.vacuum_on = True
        self.last_vacuum_toggle_step = 0
        self.total_timesteps = 0
        self.counter = 0
        self.recording = False
        self.gripper_state = 0
        self.gripper_state_change_time = time.time()  # Initialize to current time
        self.gripper_on = False
        self.gripper_off = False
        self.teachbot_positions = deque()
        self.ee_dof_states = [0.0] * self.dof_ee  # Track all EE DOF states

        # Connection elements
        self.sock = None
        self.recv_sock = None
        self.connected = False
        self.play_recording_active = False

        # Threads & flags
        self.status_streaming = False
        self.receive_target_pos = False
        self.robot_running = False
        self.listener_running = False
        self.status_thread = None
        self.update_target_info_thread = None
        self.robot_control_thread = None
        self.listener_thread = None

        # Kinematic states
        self.target_pos_received = None
        self.joint_state_received = None

        # Speed limiter state for trajectory planning
        self.previous_joint_position = None
        self.previous_timestamp = None

        # Playback speed ratio tracking (for waypoint adjustment)
        self.playback_speed_ratio = 1.0
        self.recording_speed = self.default_recording_speed  # Speed used when recording was made

        # Connect automatically if desired:
        if not self.connect():
            raise Exception(
                "TechmanRobot: Could not connect to robot. Check IP address, connection, and robot state."
            )
        
        # Initialize end effector communication based on type
        if self.ee_type == "external":
            self.logger_ri.info(f"Initializing external EE communication with PLC, trigger function: {self.trigger_function}")
            self._initialize_plc_communication()
        else:
            self.logger_ri.info("Using integrated EE - no additional communication setup needed")
        
        # Initialize EE states from start position
        self._initialize_ee_states()
        
        # Counter for reducing PLC communication frequency
        self.plc_send_counter = 0
        self.plc_send_frequency_divisor = 10  # Send PLC data every 10th cycle

    def _initialize_ee_states(self):
        """
        Initialize end-effector states from start position or defaults.
        """
        if len(self.start_position) >= self.total_dof:
            # Extract EE states from start position
            self.ee_dof_states = self.start_position[self.dof:self.dof + self.dof_ee]
            self.logger_ri.info(f"Initialized EE states from start position: {self.ee_dof_states}")
        else:
            # Use default values (typically all zeros)
            self.ee_dof_states = [0.0] * self.dof_ee
            self.logger_ri.info(f"Initialized EE states to defaults: {self.ee_dof_states}")
            
        # Initialize gripper state from first EE DOF if available
        if self.dof_ee > 0:
            initial_gripper_state = 1 if self.ee_dof_states[0] > self.gripper_treshold else 0
            self.gripper_state = initial_gripper_state
            self.logger_ri.info(f"Initialized gripper state to: {self.gripper_state}")

    ################################################################
    # EXTERNAL EE PLC COMMUNICATION METHODS
    ################################################################

    def _initialize_plc_communication(self):
        """
        Initialize MQTT communication with PLC for external end effector control.
        """
        if not MQTT_AVAILABLE:
            self.logger_ri.error("paho-mqtt not available. Cannot initialize external EE communication.")
            self.plc_connected = False
            return

        self.logger_ri.info("Initializing MQTT communication for external end effector...")
        
        # Get MQTT configuration from config file
        mqtt_config = self.config["hardware"]["robot"].get("mqtt", {})
        self.mqtt_broker_host = mqtt_config.get("broker_host", "localhost")
        self.mqtt_broker_port = mqtt_config.get("broker_port", 1883)
        self.mqtt_topic = mqtt_config.get("topic", "TOS/ee")
        self.mqtt_client_id = mqtt_config.get("client_id", f"TechmanRobot_{int(time.time())}")
        
        # Get select value from config
        ee_config = self.config["hardware"]["robot"].get("ee_config", {})
        self.select_value = ee_config.get("select", True)
        
        # Initialize MQTT client variables
        self.plc_connected = False
        self.mqtt_client = None
        
        try:
            # Create MQTT client
            self.mqtt_client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION1, 
                client_id=self.mqtt_client_id, 
                protocol=mqtt.MQTTv311
            )
            
            # Set callbacks
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_publish = self._on_mqtt_publish
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            
            # Connect to MQTT broker
            self.logger_ri.info(f"Connecting to MQTT broker {self.mqtt_broker_host}:{self.mqtt_broker_port}")
            self.mqtt_client.connect(self.mqtt_broker_host, self.mqtt_broker_port, 60)
            self.mqtt_client.loop_start()
            
            # Wait for connection
            time.sleep(1)
            
            if self.plc_connected:
                self.logger_ri.info("MQTT communication initialized successfully")
                self.logger_ri.info(f"MQTT Config - Broker: {self.mqtt_broker_host}:{self.mqtt_broker_port}")
                self.logger_ri.info(f"MQTT Config - Topic: {self.mqtt_topic}, Client ID: {self.mqtt_client_id}")
                self.logger_ri.info(f"Select value from config: {self.select_value}")
            else:
                self.logger_ri.error("Failed to connect to MQTT broker")
            
        except Exception as e:
            self.logger_ri.error(f"Failed to initialize MQTT communication: {e}")
            self.plc_connected = False

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.plc_connected = True
            self.logger_ri.info(f"✓ Connected to MQTT broker {self.mqtt_broker_host}:{self.mqtt_broker_port}")
            self.logger_ri.info(f"✓ Client ID: {self.mqtt_client_id}")
            self.logger_ri.info(f"✓ Publishing to topic: {self.mqtt_topic}")
        else:
            self.plc_connected = False
            self.logger_ri.error(f"✗ Failed to connect to MQTT broker. Return code: {rc}")

    def _on_mqtt_publish(self, client, userdata, mid):
        """MQTT publish callback"""
        # Uncomment for debugging publish confirmations
        # self.logger_ri.debug(f"MQTT message {mid} published")
        pass

    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback"""
        self.plc_connected = False
        if rc != 0:
            self.logger_ri.warning(f"⚠ Unexpected disconnection from MQTT broker. Return code: {rc}")
        else:
            self.logger_ri.info("✓ Disconnected from MQTT broker")

    def _send_ee_data_to_plc(self, ee_data):
        """
        Send end effector data to the PLC via MQTT.
        Data is sent at reduced frequency (1/10th of control loop) to prevent PLC communication delays.
        
        Args:
            ee_data: List of end effector DOF values
            
        Returns:
            bool: True if data was sent successfully or skipped due to frequency control, False on error
        """
        if not self.plc_connected or not self.mqtt_client:
            self.logger_ri.warning("MQTT not connected - cannot send EE data")
            return False
        
        # Increment counter for frequency control
        self.plc_send_counter += 1
        
        # Send EE data to PLC only every Nth cycle to reduce frequency
        if self.plc_send_counter < self.plc_send_frequency_divisor:
            # Skip sending this cycle - return True to indicate successful handling
            return True
        
        # Reset counter - time to send data
        self.plc_send_counter = 0
            
        try:
            # Extract and scale the values
            # eedof1 (setpoint): potentiometer value 0-1, scale to 0-4095 (12-bit word)
            setpoint_raw = ee_data[0] if len(ee_data) > 0 else 0.0
            setpoint_scaled = int(setpoint_raw * 4095)  # Scale 0-1 to 0-4095
            
            # eedof2 (grind): 0/1 value, convert to boolean
            grind_raw = ee_data[1] if len(ee_data) > 1 else 0.0
            grind_value = bool(int(grind_raw))  # Convert to boolean
            
            # Create MQTT payload
            payload = {
                "Grind": grind_value,           # M101: Boolean (True/False) from eedof2
                "Select": self.select_value,    # M103: Boolean from config
                "Setpoint": setpoint_scaled     # MW 0100: Word (0-4095) from eedof1 scaled
            }
            
            # Convert to JSON
            payload_json = json.dumps(payload)
            
            # Publish to MQTT
            result = self.mqtt_client.publish(self.mqtt_topic, payload_json, qos=0)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                return True
            else:
                self.logger_ri.error(f"Failed to publish MQTT message. Return code: {result.rc}")
                return False
            
        except Exception as e:
            self.logger_ri.error(f"Error sending EE data via MQTT: {e}")
            return False

    def update_select_value_from_command(self, full_message):
        """
        Update the select value based on the extra_function1 parameter from the command.
        extra_function1 = True -> Pressure mode -> select = True
        extra_function1 = False -> Position mode -> select = False
        """
        if full_message and "extra_function1" in full_message:
            extra_function1 = full_message["extra_function1"]
            # Map extra_function1 to select value (for BR customer: Pressure=True, Position=False)
            self.select_value = bool(extra_function1)
            self.logger_ri.info(f"Updated select_value to {self.select_value} based on extra_function1: {extra_function1} ({'Pressure' if extra_function1 else 'Position'} mode)")
        else:
            self.logger_ri.debug("No extra_function1 parameter found in command, keeping current select_value")

    ################################################################
    # 1) COMMANDS
    ################################################################

    def start_teleoperation(self, full_message):
        """
        Move to the start position and start teleoperation.
        Create a thread to listen for target positions, then
        start the teleoperation control loop in another thread.
        """
        # Update select value based on extra_function1 parameter
        self.update_select_value_from_command(full_message)
        
        self.robot_running = True
        self.receive_target_pos = True

        # Ensure robot is connected
        if not self.connected:
            if not self.connect():
                self.logger_ri.error(
                    "start_teleoperation, could not connect to robot. "
                    "Check IP address, connection, or robot state."
                )
                return False

        # Start listener thread for robot feedback
        try:
            self.start_listener_thread()
        except Exception as e:
            self.logger_ri.error(f"Error starting listener thread: {e}")
            return False

        # Create thread that generates target positions
        self.update_target_info_thread = threading.Thread(
            target=self._update_target_info, args=(full_message, ), daemon=True
        )
        self.update_target_info_thread.start()
        self.logger_ri.info("start_teleoperation, target position updating started.")

        # create thread that controls the robot
        self.robot_control_thread = threading.Thread(
            target=self._control_robot, args=(full_message, ), daemon=True
        )
        self.robot_control_thread.start()
        self.logger_ri.info("start_teleoperation, robot control thread created.")
        return True

    def start_teleoperation_record(self, full_message=None):
        """
        Same as start_teleoperation, but also sets self.recording=True
        so we store positions in shm_joint_data1.
        """
        self.recording = True
        return self.start_teleoperation(full_message)

    def record_episodes(self, full_message=None):
        self.recording = True
        return self.start_teleoperation(full_message)

    def run_policy(self, full_message=None):
        """
        Run the policy execution.
        """
        if full_message is None:
            full_message = {"message": "run_policy"}
        self.robot_running = True
        self.run_policy_active = True

        # Ensure robot is connected
        if not self.connected:
            if not self.connect():
                self.logger_ri.error(
                    "run_policy, could not connect to robot. "
                    "Check IP address, connection, or robot state."
                )
            self.logger_ri.error("run_policy, robot not connected.")
            return False

        # Set robot speed for policy execution from model config
        model_name = full_message.get('model_name')
        dataset_name = full_message.get('dataset_name')
        if model_name and dataset_name:
            dataset_dir = get_data_path(self.config, dataset_name)
            model_config_path = os.path.join(dataset_dir, "Models", model_name, 'config.json')
                    
            if os.path.exists(model_config_path):
                with open(model_config_path, 'r') as f:
                    model_config = json.load(f)
                self.robot_speed = model_config.get('robot_speed', self.default_recording_speed)
                self.logger_ri.info(f"Using robot speed from model config: {self.robot_speed}")
            else:
                self.logger_ri.warning(f"Model config not found at {model_config_path}, using default robot speed")
                self.robot_speed = self.default_recording_speed
        else:
            self.robot_speed = self.default_recording_speed

        # create thread that controls the robot
        self.robot_control_thread = threading.Thread(
            target=self._control_robot, args=(full_message, ), daemon=True
        )
        self.robot_control_thread.start()
        self.logger_ri.info("run_policy, robot control thread created for policy execution.")
        return True

    def play_recording(self, full_message=None):
        """
        Play the recorded positions from a JSON file.
        """
        if full_message is None:
            full_message = {"message": "play_recording"}
            
        # Update select value based on extra_function1 parameter
        self.update_select_value_from_command(full_message)
            
        self.robot_running = True
        self.play_recording_active = True

        # Ensure robot is connected
        if not self.connected:
            if not self.connect():
                self.logger_ri.error(
                    "play_recording, could not connect to robot. "
                    "Check IP address, connection, or robot state."
                )
                return False

        # Start listener thread for robot feedback
        try:
            self.start_listener_thread()
        except Exception as e:
            self.logger_ri.error(f"Error starting listener thread: {e}")
            return False

        # Create thread that generates target positions
        self.update_target_info_thread = threading.Thread(
            target=self._update_target_info, args=(full_message, ), daemon=True
        )
        self.update_target_info_thread.start()
        self.logger_ri.info("play_recording, target position updating started.")

        # create thread that controls the robot
        self.robot_control_thread = threading.Thread(
            target=self._control_robot, args=(full_message, ), daemon=True
        )
        self.robot_control_thread.start()
        self.logger_ri.info("play_recording, robot control thread created.")
        return True

    def stop(self):
        """
        Stop teleoperation or recording if running,
        stop UDP streaming, then disconnect from robot.
        """
        if not self.connected:
            self.logger_ri.info("stop, already disconnected.")
            return True

        # Stop robot if running
        self.logger_ri.info("stop, stopping robot running.")
        if self.robot_running:
            self.stop_robot_running()
        self.logger_ri.info("stop, robot running stopped.")

        # Stop listener thread
        self.logger_ri.info("stop, stopping listener thread.")
        try:
            self.stop_listener_thread()
        except Exception as e:
            self.logger_ri.error(f"Error stopping listener thread: {e}")

        # Stop status streaming
        self.logger_ri.info("stop, stopping status streaming.")
        self.stop_status_streaming()

        # Disconnect
        self.logger_ri.info("stop, disconnecting.")
        success = self.disconnect()
        
        # Reset state flags
        self.recording = False
        self.play_recording_active = False
        self.run_policy_active = False
        
        self.logger_ri.info("stop, disconnected.")
        return success

    def disconnect(self):
        """
        Disconnect from the Techman robot and clean up MQTT connection.
        """
        if not self.connected:
            self.logger_ri.warning("Already disconnected.")
            return True

        # Close MQTT connection if external EE is used
        if self.ee_type == "external" and hasattr(self, 'plc_connected') and self.plc_connected:
            try:
                if hasattr(self, 'mqtt_client') and self.mqtt_client:
                    self.mqtt_client.loop_stop()
                    self.mqtt_client.disconnect()
                    self.mqtt_client = None
                self.plc_connected = False
                self.logger_ri.info("Disconnected from MQTT broker")
            except Exception as e:
                self.logger_ri.error(f"Error disconnecting from MQTT broker: {e}")

        # Close robot sockets
        try:
            if self.sock:
                self.sock.close()
                self.sock = None
        except Exception as e:
            self.logger_ri.error(f"Error closing main socket: {e}")

        self.connected = False
        self.logger_ri.info("Disconnected from Techman.")
        return True

    ################################################################
    # 2) CORE ROBOT CONTROL METHODS
    ################################################################

    def _update_target_info(self, full_message):
        """
        Thread that continuously reads target positions from shared memory
        or loads recording data for playback.
        """
        if self.play_recording_active:
            if not self._load_recording_data(full_message):
                return  # Stop if loading failed
        else:
            self._receive_target_pos_loop()

    def _load_recording_data(self, full_message):
        """
        Load recorded positions from file for playback.
        """
        try:
            filename = full_message["recording_name"]
            filename = get_data_path(self.config, filename, False)
            if not os.path.exists(filename):
                self.logger_ri.error(f"play_recording, file {filename} not found.")
                self.stop_robot_running()
                return False
        except KeyError:
            self.logger_ri.error("_load_recording_data, missing recording_name in message.")
            self.stop_robot_running()
            return False

        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            if "samples" not in data or len(data["samples"]) == 0:
                self.logger_ri.error(f"No samples in file with name: {filename}")
                self.stop_robot_running()
                return False

            # Get recording speed from metadata or use default
            self.recording_speed = data.get("recording_speed", self.default_recording_speed)
            
            # Get playback speed from message or use default
            playback_speed = full_message.get("playback_speed", "")
            
            if playback_speed == "":
                self.playback_speed_ratio = 1.0
                self.logger_ri.info("_load_recording_data, no playback speed specified, using ratio 1.0")
            else:
                try:
                    playback_speed_val = float(playback_speed)
                    # Calculate effective speed and ratio
                    effective_speed = playback_speed_val * self.recording_speed
                    
                    if effective_speed > 1.0:
                        effective_speed = 1.0  # Cap at maximum robot speed
                        self.logger_ri.info(f"_load_recording_data, effective speed capped at 1.0 (was {playback_speed_val * self.recording_speed:.2f})")
                    
                    # Calculate speed ratio for waypoint adjustment
                    self.playback_speed_ratio = effective_speed / self.recording_speed
                    
                    self.logger_ri.info(f"_load_recording_data, recording speed: {self.recording_speed}, "
                                      f"playback speed: {playback_speed_val}, effective speed: {effective_speed}, "
                                      f"speed ratio: {self.playback_speed_ratio:.2f}")
                except ValueError:
                    self.logger_ri.error(f"_load_recording_data, invalid playback speed: {playback_speed}")
                    self.playback_speed_ratio = 1.0

            # Load positions
            self.teachbot_positions = deque([s["teachbot_position"] for s in data["samples"]])
            
            # Adjust waypoints based on speed ratio
            self._adjust_teachbot_positions_for_playback_speed()
            
            try:
                self.first_joint_position = data["samples"][0]["robot_position"]
            except IndexError:
                self.logger_ri.error(f"_load_recording_data, no robot positions found in {filename}.")
                self.stop_robot_running()
                return False
            
            # Safety check for starting position
            if not np.all(np.abs(np.array(self.first_joint_position) - np.array(self.start_position)) < 5):
                self.logger_ri.error(
                    f"First joint position is not within 5 degrees of the start position. "
                    f"Start position: {self.start_position}, First joint position: {self.first_joint_position}"
                )
                self.stop_robot_running()
                return False
                
        except Exception as e:
            self.logger_ri.error(f"Error loading recording data: {e}")
            self.stop_robot_running()
            return False
            
        return True

    def _adjust_teachbot_positions_for_playback_speed(self):
        """
        Adjust teachbot positions based on playback speed ratio.
        
        - Ratio > 1: Skip waypoints (use every Nth waypoint)
        - Ratio = 1: No adjustment needed 
        - Ratio < 1: Interpolate additional waypoints
        """
        if abs(self.playback_speed_ratio - 1.0) < 0.01:
            # No adjustment needed for ratio ~1.0
            self.logger_ri.info(f"_adjust_teachbot_positions_for_playback_speed, ratio {self.playback_speed_ratio:.2f} ≈ 1.0, no adjustment needed. Waypoint count: {len(self.teachbot_positions)}")
            return
            
        original_count = len(self.teachbot_positions)
        original_positions = list(self.teachbot_positions)
        
        if self.playback_speed_ratio > 1.0:
            # Skip waypoints - use every Nth waypoint where N = ratio
            skip_factor = int(round(self.playback_speed_ratio))
            adjusted_positions = original_positions[::skip_factor]
            
            self.logger_ri.info(f"_adjust_teachbot_positions_for_playback_speed, ratio {self.playback_speed_ratio:.2f} > 1.0, skipping waypoints with factor {skip_factor}")
            self.logger_ri.info(f"Waypoint count: {original_count} -> {len(adjusted_positions)} (reduction: {((original_count - len(adjusted_positions)) / original_count * 100):.1f}%)")
            
        else:  # self.playback_speed_ratio < 1.0
            # Interpolate additional waypoints
            interpolation_factor = int(round(1.0 / self.playback_speed_ratio))
            adjusted_positions = []
            
            self.logger_ri.info(f"_adjust_teachbot_positions_for_playback_speed, ratio {self.playback_speed_ratio:.2f} < 1.0, interpolating waypoints with factor {interpolation_factor}")
            
            for i in range(len(original_positions)):
                adjusted_positions.append(original_positions[i])
                
                # Interpolate between current and next position (if next exists)
                if i < len(original_positions) - 1:
                    current_pos = np.array(original_positions[i])
                    next_pos = np.array(original_positions[i + 1])
                    
                    # Create intermediate positions
                    for j in range(1, interpolation_factor):
                        alpha = j / interpolation_factor
                        # Initialize interpolated position array
                        interpolated_pos = np.zeros_like(current_pos)
                        
                        # Interpolate each joint angle with wraparound handling
                        for joint_idx in range(len(current_pos)):
                            # Get the two angles
                            angle1 = current_pos[joint_idx]
                            angle2 = next_pos[joint_idx]
                            
                            # Calculate shortest angular distance with wraparound
                            diff = angle2 - angle1
                            # Normalize to [-180, 180]
                            if diff > 180:
                                diff -= 360
                            elif diff < -180:
                                diff += 360
                                
                            # Calculate interpolated angle
                            interpolated_angle = angle1 + alpha * diff
                            # Normalize result to [-180, 180]
                            if interpolated_angle > 180:
                                interpolated_angle -= 360
                            elif interpolated_angle < -180:
                                interpolated_angle += 360
                                
                            interpolated_pos[joint_idx] = interpolated_angle
                            
                        adjusted_positions.append(interpolated_pos.tolist())
            
            self.logger_ri.info(f"Waypoint count: {original_count} -> {len(adjusted_positions)} (increase: {((len(adjusted_positions) - original_count) / original_count * 100):.1f}%)")
        
        # Replace the original positions with adjusted ones
        self.teachbot_positions = deque(adjusted_positions)
        self.logger_ri.info(f"_adjust_teachbot_positions_for_playback_speed, waypoint adjustment completed")

    def _receive_target_pos_loop(self):
        """
        Thread that continuously reads from shm_target_pos1
        """
        # First empty the queue
        while not self.shm_target_pos1.empty():
            try:
                pos = self.shm_target_pos1.get_nowait()
            except Exception:
                pass

        while self.receive_target_pos:
            try:
                target_pos = self.shm_target_pos1.get(timeout=0.1)
                self.target_pos_received = target_pos

            except Exception as e:
                pass
            time.sleep(self.control_dt / self.check_queue_period_divisor)

    def _control_robot(self, full_message):
        """
        Main control thread for teleoperation or playback.
        Follows the same structure as Fanuc robot:

        1) Move to start position.
        2) Check start position tolerance or relevant conditions.
        3) Prepare robot (enable continuous mode, setup streaming).
        4) Control loop (main teleoperation or playback loop).
        5) Close robot streaming.
        6) Send 'stop' response if in playback mode.
        """
        # move to start position
        self._move_to_start_position()

        # check if start position is alright
        self._check_start_position(full_message)

        # prepare robot
        started_streaming = self._prepare_robot(full_message)

        # control loop (trajectory, buffer fill, etc.)
        self._control_loop(started_streaming)

        # close robot
        self._close_robot(started_streaming)

        # if in playback mode, send "stop" after finishing
        self._send_stop_response(full_message)

    def _move_to_start_position(self):
        """
        Move robot to 'start_position' (from config).
        """
        if not self.move_to_start_position():
            self.logger_ri.error("_move_to_start_position, could not move to start position.")
            return False
        self.logger_ri.info("_move_to_start_position, moved to start position.")
        return True

    def _check_start_position(self, full_message):
        """
        Verify that the current position is near the start position,
        or handle 'opening_ceremony' if teleoperation is started.
        """
        message = full_message.get("message", "")

        if message == "start_teleoperation" or message == "start_teleoperation_record" or message == "record_episodes":
            self.logger_ri.info("checking start position")
            if not self.opening_ceremony():
                self.logger_ri.error("_check_start_position, opening ceremony failed")
                return False

        if message == "play_recording":
            # Check if start position is close to the first recording position
            if hasattr(self, 'first_joint_position'):
                diffs = [abs(c - s) for c, s in zip(self.start_position, self.first_joint_position)]
                if any(d > self.start_joint_tolerance for d in diffs):
                    self.logger_ri.error(
                        "_check_start_position, start position is not close to the first position in the recording"
                    )
                    return False
            self.logger_ri.info("_check_start_position, checking start position")
            return True

    def _prepare_robot(self, full_message):
        """
        Make the robot ready for streaming:
        - Enable continuous mode on Techman
        - Setup any required streaming parameters
        """
        started_streaming = False
        if self.robot_running:
            # Enable continuous mode on Techman for real-time commands
            speed_limit_cmd = 'SetTCPSpeedLimit(true, 0, 1)'
            send_tmsct_cmd(self.sock, "1", speed_limit_cmd, self.logger_ri)
            enable_cmd = f'Position(true, "J", {self.continuous_accel_time}, {self.continuous_motion_gain}, {self.continuous_protection_time})'
            send_tmsct_cmd(self.sock, "1", enable_cmd, self.logger_ri)
            
            self.logger_ri.info("_prepare_robot, enabled continuous mode")
            started_streaming = True

        return started_streaming

    def _control_loop(self, started_streaming):
        """
        Core loop for either teleoperation or playing back a recorded trajectory:
        - Setup motion planner with speed limiting
        - Fill action buffer with start position
        - While running, compute next motion and send to robot
        - Optionally store motion data if recording
        """
        # Set up motion planner with speed limiting and initial conditions
        self.logger_ri.info("control_loop, Setup motion planner with speed limiting")
        otg, inp, out = self.setup_planner()
        self.logger_ri.info("control_loop, Motion planner setup done.")

        # Garbage collection optimization for real-time performance
        self.logger_ri.info("control_loop, Starting GC optimization")
        import gc
        # Store original GC settings to restore later
        original_gc_thresholds = gc.get_threshold()
        
        # Set optimized GC thresholds for real-time performance
        # Higher thresholds = less frequent GC = more predictable timing
        optimized_gen0 = 2000   # Less frequent gen0 (default: 700)
        optimized_gen1 = 25     # Less frequent gen1 (default: 10)
        optimized_gen2 = 25     # Less frequent gen2 (default: 10)
        gc.set_threshold(optimized_gen0, optimized_gen1, optimized_gen2)
        self.logger_ri.info("control_loop, GC optimization done")
        
        time.sleep(0.01)
        self.logger_ri.info("control_loop, After sleep, starting buffer fill")

        self.logger_ri.info("control_loop, starting control loop")
        previous_action_master = self.start_position

        self.logger_ri.info("control_loop, entering main while loop")
        while self.robot_running:
            start_time = time.time()
            
            # Get robot feedback - Techman uses start position as current state
            last_received_time = time.time()
            last_received_js = self.start_position

            # Get next action
            if self.play_recording_active:
                if self.teachbot_positions:
                    action_master = self.teachbot_positions.popleft()
                else:
                    self.logger_ri.info("control_loop, recording playback completed")
                    break
            elif self.run_policy_active:
                action_master = self._get_next_policy_action()
                if action_master is None:
                    action_master = previous_action_master
            else:
                if self.target_pos_received is not None:
                    action_master = self.target_pos_received
                else:
                    # No action received, use previous action
                    action_master = previous_action_master

            # Extract EE DOF values from action_master
            self.ee_dof_states = self.extract_ee_dof_values(action_master)
            
            # Trajectory calculation with joint speed limiting (robot joints only)
            self.update_input(action_master, inp, previous_action_master)
            previous_action_master = action_master

            success_calc = self.trajectory_calculation(otg, inp, out)
            
            # Use first EE DOF for gripper state determination
            if self.dof_ee > 0:
                self.determine_gripper_state(self.ee_dof_states[0])
            else:
                # Fallback for backward compatibility
                if len(action_master) > self.dof:
                    self.determine_gripper_state(action_master[-1])
            
            if not success_calc:
                self.logger_ri.error("control_loop, trajectory calculation failed")
                break

            sent_robot_position = out["new_position"]
            teachbot_position = inp["target_position"]

            # Store flags before sending
            gripper_on_to_send = self.gripper_on
            gripper_off_to_send = self.gripper_off
            
            # Send to robot via TMSCT
            success_send = self._send_robot_position(sent_robot_position, gripper_on_to_send, gripper_off_to_send)
            
            # Reset gripper flags after sending
            self.gripper_on = False
            self.gripper_off = False
            
            if not success_send:
                self.logger_ri.error("control_loop, could not send joint position to robot")
                break
            
            # Upload data to shm_joint_data1 if recording
            if self.recording:
                try:
                    if self.shm_joint_data1.full():
                        self.shm_joint_data1.get_nowait()
                    
                    # Ensure positions are lists and append all EE DOF states
                    teachbot_pos_list = list(teachbot_position) if not isinstance(teachbot_position, list) else teachbot_position
                    sent_pos_list = list(sent_robot_position) if not isinstance(sent_robot_position, list) else sent_robot_position
                    last_received_list = list(last_received_js) if not isinstance(last_received_js, list) else last_received_js
                    
                    teachbot_pos_with_ee = teachbot_pos_list + self.ee_dof_states
                    sent_pos_with_ee = sent_pos_list + self.ee_dof_states
                    # last_received_js already contains EE states as last elements
                    
                    joint_data = {
                        "teachbot_position": teachbot_pos_with_ee,
                        "sent_robot_position": sent_pos_with_ee,
                        "robot_position": last_received_list,  # Already includes EE states
                        "robot_position_timestamp": last_received_time,
                        "seq_id": getattr(self, 'seq_id', 0),
                    }
                    self.shm_joint_data1.put(joint_data)
                except Exception as e:
                    self.logger_ri.error("control_loop, error uploading joint data: %s", e)

            if self.run_policy_active:
                try:
                    if self.shm_joint_data2.full():
                        self.shm_joint_data2.get_nowait()
                    
                    # Append all EE DOF states to position arrays
                    teachbot_pos_with_ee = teachbot_position + self.ee_dof_states
                    sent_pos_with_ee = sent_robot_position + self.ee_dof_states
                    
                    joint_data = {
                        "teachbot_position": teachbot_pos_with_ee,
                        "sent_robot_position": sent_pos_with_ee,
                        "robot_position": last_received_js,
                        "robot_position_timestamp": last_received_time,
                        "seq_id": getattr(self, 'seq_id', 0),
                    }
                    self.shm_joint_data2.put(joint_data)
                except Exception as e:
                    self.logger_ri.error("control_loop, error uploading joint data: %s", e)

            # Rate control - maintain 100Hz control loop
            elapsed = time.time() - start_time
            sleep_time = self.control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Restore original GC settings when exiting
        gc.set_threshold(*original_gc_thresholds)

        if started_streaming:
            self.logger_ri.info("control_loop, stopping robot")
            self.stop_robot(otg, inp, out)
            self.logger_ri.info("control_loop, robot stopped")

    def _close_robot(self, started_streaming):
        """
        Close robot streaming and disable continuous mode.
        """
        if started_streaming:
            # Disable continuous mode on Techman
            disable_cmd = f'Position(false, "J", {self.continuous_accel_time}, {self.continuous_motion_gain}, {self.continuous_protection_time})'
            send_tmsct_cmd(self.sock, "3", disable_cmd, self.logger_ri)
            self.logger_ri.info("_close_robot, disabled continuous mode")

    def _send_stop_response(self, full_message):
        """
        If the robot was playing a recording, send a 'stop' command
        back to the queue to indicate the operation completed.
        """
        message = full_message.get("message", "")
        if message == "play_recording":
            send_tc_command(self.robot_interface_commup, {"type": "CMD", "message": "stop", "interface": self.component_tag})
            self.logger_ri.info("_send_stop_response, sent stop command to queue.")

    ################################################################
    # MOTION PLANNING FUNCTIONS
    ################################################################

    def setup_planner(self):
        """
        Set up motion planner with initial conditions.
        Keep everything in teachbot coordinates - translation happens only at robot command level.
        """
        
        # Reset speed limiter state
        self.previous_joint_position = None
        self.previous_timestamp = None
        
        # Keep start position in teachbot coordinates for trajectory planner
        start_position_teachbot = self.start_position[:self.dof]
        
        # Create simple dictionary-based objects to mimic Ruckig interface
        otg = {
            "type": "motion_planner",
            "velocity_limits": self.safe_velocity_limits.copy()
        }
        
        inp = {
            "current_position": start_position_teachbot,
            "current_velocity": [0.0] * self.dof,
            "current_acceleration": [0.0] * self.dof,
            "target_position": start_position_teachbot,
            "target_velocity": [0.0] * self.dof,
            "target_acceleration": [0.0] * self.dof,
        }
        
        out = {
            "new_position": start_position_teachbot,
            "new_velocity": [0.0] * self.dof,
            "new_acceleration": [0.0] * self.dof,
        }
        
        self.logger_ri.info(f"Motion planner setup with velocity limits: {self.safe_velocity_limits}")
        return otg, inp, out

    def update_input(self, action_master, inp, previous_action_master):
        """
        Process the new action (master) against position limits and
        set the input accordingly.
        Keep everything in teachbot coordinates - translation happens only when sending to robot.
        """
        current_position = [round(pos, 6) for pos in inp["current_position"]]
        inp["current_position"] = current_position

        # Keep everything in teachbot coordinates for path planning
        raw_target_position = action_master[:self.dof].copy()
        
        # Apply position limits in teachbot coordinates
        # Note: We need to translate limits to teachbot coordinate system for proper checking
        target_position = raw_target_position.copy()
        
        # For now, use basic range checking - you might want to implement proper limit translation
        for i in range(self.dof):
            # These are rough teachbot coordinate limits - adjust as needed
            teachbot_upper_limits = [180, 180, 180, 180, 180, 180]
            teachbot_lower_limits = [-180, -180, -180, -180, -180, -180]
            
            if target_position[i] > teachbot_upper_limits[i]:
                self.logger_ri.warning(f"Joint {i} target {target_position[i]:.1f} exceeds upper limit {teachbot_upper_limits[i]}, clamping")
                target_position[i] = teachbot_upper_limits[i]
            elif target_position[i] < teachbot_lower_limits[i]:
                self.logger_ri.warning(f"Joint {i} target {target_position[i]:.1f} below lower limit {teachbot_lower_limits[i]}, clamping")
                target_position[i] = teachbot_lower_limits[i]

        inp["target_position"] = target_position
        return

    def trajectory_calculation(self, otg, inp, out):
        """
        Trajectory calculation with joint speed limiting.
        Implements velocity constraints based on config velocity_limits and safety factor.
        """
        try:
            current_position = inp["current_position"]
            target_position = inp["target_position"]
            current_time = time.time()
            
            # Initialize previous state if not set
            if self.previous_joint_position is None or self.previous_timestamp is None:
                self.previous_joint_position = current_position.copy()
                self.previous_timestamp = current_time
                dt = self.control_dt  # Use control timestep for first iteration
            else:
                # Calculate time step
                dt = current_time - self.previous_timestamp
                if dt <= 0:
                    dt = self.control_dt  # Fallback to control timestep
            
            # Apply speed limits to each joint
            limited_position = []
            actual_velocities = []
            
            for i in range(self.dof):
                # Calculate required position change with angular wraparound handling
                raw_position_diff = target_position[i] - current_position[i]
                
                # Handle angular wraparound to ensure shortest path
                if abs(raw_position_diff) > 180:
                    if raw_position_diff > 0:
                        position_diff = raw_position_diff - 360
                    else:
                        position_diff = raw_position_diff + 360
                else:
                    position_diff = raw_position_diff
                
                # Calculate maximum allowed position change based on velocity limit
                max_velocity = self.safe_velocity_limits[i]  # degrees/second with safety factor
                max_position_change = max_velocity * dt
                
                # Limit the position change if necessary
                if abs(position_diff) > max_position_change:
                    # Scale down the position change to respect velocity limit
                    if position_diff > 0:
                        limited_change = max_position_change
                    else:
                        limited_change = -max_position_change
                    
                    new_position = current_position[i] + limited_change
                    actual_velocity = limited_change / dt
                    
                else:
                    # No limiting needed
                    new_position = current_position[i] + position_diff
                    actual_velocity = position_diff / dt
                
                limited_position.append(new_position)
                actual_velocities.append(actual_velocity)
            
            # Set output
            out["new_position"] = limited_position
            out["new_velocity"] = actual_velocities
            out["new_acceleration"] = [0.0] * self.dof  # Placeholder
            
            # Update state for next iteration
            self.previous_joint_position = out["new_position"].copy()
            self.previous_timestamp = current_time
            
            # Update input for next iteration
            inp["current_position"] = out["new_position"].copy()
            inp["current_velocity"] = out["new_velocity"].copy()
            inp["current_acceleration"] = out["new_acceleration"].copy()
            
            return True
            
        except Exception as e:
            self.logger_ri.error(f"Trajectory calculation error: {e}")
            return False

    def stop_robot(self, otg, inp, out):
        """
        Robot stop function.
        Sends a few more position commands to ensure robot stops.
        Translates from teachbot to Techman coordinates before sending.
        """
        self.logger_ri.info("stop_robot, stopping robot")
        nr_stop_actions = 50
        stop_step_counter = 0

        while stop_step_counter < nr_stop_actions:
            # Get current position in teachbot coordinates and translate to Techman
            teachbot_position = out["new_position"].copy()
            translated_position = action_master_TM_translation(teachbot_position, self.logger_ri)
            script_cmd = "Position({:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f})".format(*translated_position)
            
            send_tmsct_cmd(self.sock, "2", script_cmd, self.logger_ri)
            
            stop_step_counter += 1
            time.sleep(self.control_dt)
            
        self.logger_ri.info("stop_robot, done stopping robot")
        return True

    def _send_robot_position(self, robot_position, gripper_on, gripper_off):
        """
        Send robot position to Techman via TMSCT protocol.
        Translate from teachbot coordinates to Techman coordinates right before sending.
        """
        try:
            # robot_position is in teachbot coordinates - translate to Techman coordinates
            teachbot_position = robot_position.copy()
            translated_position = action_master_TM_translation(teachbot_position, self.logger_ri)

            script_cmd = "Position({:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f})".format(*translated_position)
                  
            # Send command to robot
            success = send_tmsct_cmd(self.sock, "2", script_cmd, self.logger_ri)
            
            # Handle external EE communication - frequency control is now handled in _send_ee_data_to_plc
            if self.ee_type == "external":
                plc_success = self._send_ee_data_to_plc(self.ee_dof_states)
                if not plc_success:
                    self.logger_ri.warning("Failed to send EE data to PLC")
            
            return success
            
        except Exception as e:
            self.logger_ri.error(f"Error sending robot position: {e}")
            return False

    def _get_next_policy_action(self):
        """
        Placeholder for policy action retrieval.
        Returns None since policy execution not yet implemented for Techman.
        """
        self.logger_ri.warning("Policy execution not yet implemented for Techman robot")
        return None

    ################################################################
    # LEGACY CONTROL METHODS (kept for compatibility)
    ################################################################

    def opening_ceremony(self):
        """
        Wait until the externally-provided target is within tolerance of the start position.
        """
        ready_for_teleoperation = False
        self.target_pos_received = None
        time.sleep(0.01)
        
        self.logger_ri.info("Opening ceremony: waiting for target position within tolerance")

        count = 0
        freq = int(1/self.control_dt)
        
        while self.robot_running:
            if not ready_for_teleoperation:
                if self.target_pos_received is not None:
                    target_pos_received = self.target_pos_received[:self.dof]
                    start_position = self.start_position[:self.dof]
                    
                diffs = []
                for c, s in zip(target_pos_received, start_position):
                    # Calculate the shortest angular distance between two angles
                    diff = abs(c - s)
                    # Handle wraparound for rotational joints (0-360° or -180° to 180°)
                    if diff > 180:
                        diff = 360 - diff
                    diffs.append(diff)
  
                count += 1
                if count % freq == 0:
                    self.logger_ri.info(f"Angular diffs to start position: {diffs}")
                    

                    if all(d < self.start_joint_tolerance for d in diffs):
                        ready_for_teleoperation = True
                        self.logger_ri.info("Opening ceremony: position within tolerance, starting teleoperation")
                        return True
                time.sleep(self.control_dt)
        return False


    def _execute_play_recording(self):
        """
        Execute playback of recorded positions.
        """
        self.logger_ri.info("Starting play recording control loop.")
        previous_action_master = self.start_position
        
        while self.robot_running:
            start_time = time.time()
            
            if self.teachbot_positions:
                action_deg = self.teachbot_positions.popleft()
                action_master = action_deg
            else:
                self.logger_ri.info("control_loop, recording playbook completed")
                break

            # CRITICAL FIX: Apply translation BEFORE safety check
            # Raw action_deg is in teachbot coordinates, need to translate to Techman coordinates for safety check
            translated_robot_joints = action_master_TM_translation(copy.deepcopy(action_deg[:self.dof]), self.logger_ri)
            
            self.logger_ri.debug(f"Playback translation: {action_deg[:self.dof]} -> {translated_robot_joints}")
            
            # Safety check on translated positions (in Techman coordinate system)
            if not action_master_safety_check(translated_robot_joints, self.upper_limits, self.lower_limits):
                self.logger_ri.error(f"Safety check failed for translated position: {translated_robot_joints}")
                self.logger_ri.error(f"Original teachbot position was: {action_deg[:self.dof]}")
                self.logger_ri.error(f"Upper limits: {self.upper_limits}")
                self.logger_ri.error(f"Lower limits: {self.lower_limits}")
                break

            # Extract EE DOF values from action_master
            self.ee_dof_states = self.extract_ee_dof_values(action_master)
            
            # Use first EE DOF for gripper state determination
            if self.dof_ee > 0:
                self.determine_gripper_state(self.ee_dof_states[0])
            else:
                # Fallback for backward compatibility
                if len(action_master) > self.dof:
                    self.determine_gripper_state(action_master[-1])

            # Send TMSCT command using already translated positions
            position = translated_robot_joints
            self.logger_ri.debug(f"Sending playback position command: {position}")
            script_cmd = "Position({:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f})".format(*position)

            
            if not send_tmsct_cmd(self.sock, "2", script_cmd, self.logger_ri):
                self.logger_ri.error("Error sending command to Techman.")
                break

            # Log gripper state changes instead of sending to robot
            if self.gripper_on:
                self.logger_ri.info(f"GRIPPER COMMAND: Turn ON - EE states: {self.ee_dof_states}")
                self.gripper_on = False  
            if self.gripper_off:
                self.logger_ri.info(f"GRIPPER COMMAND: Turn OFF - EE states: {self.ee_dof_states}")
                self.gripper_off = False  

            self.logger_ri.info(f"Sent recorded position command: {script_cmd}, EE states: {self.ee_dof_states}")

            previous_action_master = action_master

            # Rate control
            elapsed = time.time() - start_time
            sleep_time = self.control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.stop_robot_running(script_cmd)

    def stop_robot_running(self, script_cmd=None):  
        """
        Stop the robot control and target position receiving.
        """

        if script_cmd:
            for i in range(50):
                send_tmsct_cmd(self.sock, "2", script_cmd, self.logger_ri)
                # rate control
                start_time = time.time()
                elapsed = time.time() - start_time
                sleep_time = self.control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        self.receive_target_pos = False
        self.robot_running = False
        self.play_recording_active = False
        self.run_policy_active = False
        
        # Reset gripper states
        self.gripper_on = False
        self.gripper_off = False
        self.gripper_delay = 0

        # Join threads
        if self.update_target_info_thread and threading.current_thread() != self.update_target_info_thread:
            self.logger_ri.info("Stopping target info thread.")
            self.update_target_info_thread.join(timeout=2.0)
            self.update_target_info_thread = None

        if self.robot_control_thread and threading.current_thread() != self.robot_control_thread:
            self.logger_ri.info("Stopping robot control thread.")
            self.robot_control_thread.join(timeout=2.0)
            self.robot_control_thread = None

        return True

    ################################################################
    # 3) CONNECTION MANAGEMENT
    ################################################################

    def connect(self) -> bool:
        """
        Connect to the Techman robot via TMSCT TCP connection.
        """
        if self.connected:
            self.logger_ri.info("Already connected to Techman.")
            return True

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect(self.robot_address)
            self.connected = True
            self.logger_ri.info(f"Connected to Techman at {self.robot_address[0]}:{self.robot_address[1]}")
            
            # Start status streaming
            self.start_status_streaming()
            
            return True
        except (ConnectionRefusedError, socket.error) as e:
            self.logger_ri.error(f"Could not connect to Techman: {e}")
            if self.sock:
                self.sock.close()
                self.sock = None
            self.connected = False
            return False

    def move_to_start_position(self):
        """
        Move the robot to the start position using TMSCT commands.
        """
        if not self.connected:
            return False

        # Enable continuous mode for movement
        enable_cmd = f'Position(true, "J", {self.move_accel_time}, {self.move_motion_gain}, {self.move_protection_time})'
        send_tmsct_cmd(self.sock, "1", enable_cmd, self.logger_ri)

        # Extract only robot joints from start position for movement command
        robot_start_position = copy.deepcopy(self.start_position[:self.dof])
        position = action_master_TM_translation(robot_start_position, self.logger_ri)
        
        # Initialize EE states from start position if available
        if len(self.start_position) >= self.total_dof:
            self.ee_dof_states = self.start_position[self.dof:self.dof + self.dof_ee]
        else:
            self.ee_dof_states = [0.0] * self.dof_ee
            
        self.logger_ri.info(f"Moving to start position with {len(robot_start_position)} joints")
        
        move_time = 5
        time_now = 0

        while time_now < move_time:
            pos_cmd = "Position({:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f})".format(*position)
            
            if not send_tmsct_cmd(self.sock, "2", pos_cmd, self.logger_ri):
                self.logger_ri.error("Error sending command to Techman.")
                return False
            time_now += 0.01
            time.sleep(0.01)

        # Disable continuous mode
        disable_cmd = f'Position(false, "J", {self.move_accel_time}, {self.move_motion_gain}, {self.move_protection_time})'
        send_tmsct_cmd(self.sock, "3", disable_cmd, self.logger_ri)

        self.logger_ri.info("Completed move to start position")
        return True

    ################################################################
    # 4) STATUS STREAMING
    ################################################################

    def start_status_streaming(self):
        """
        Start status streaming thread.
        """
        if self.status_streaming:
            return
        self.status_streaming = True
        self.status_thread = threading.Thread(
            target=self.status_updating_thread, daemon=True
        )
        self.status_thread.start()

    def stop_status_streaming(self):
        """
        Stop status streaming thread.
        """
        self.status_streaming = False
        if self.status_thread:
            self.status_thread.join(timeout=1.0)
            self.status_thread = None

    def status_updating_thread(self):
        """
        Periodic status updates and monitoring.
        """
        while self.status_streaming:
            if self.connected:
                # Update status information
                pass
            time.sleep(self.status_refresh_period)

    ################################################################
    # 5) LISTENER THREAD FOR ROBOT FEEDBACK
    ################################################################

    def start_listener_thread(self):
        """
        Start UDP listener thread for robot feedback.
        """
        if self.listener_running:
            return
        self.listener_running = True
        self.listener_thread = threading.Thread(target=self._listener_loop, daemon=True)
        self.listener_thread.start()

    def stop_listener_thread(self):
        """
        Stop the listener thread.
        """
        if not self.listener_running:
            return
        self.listener_running = False
        if self.listener_thread is not None:
            self.listener_thread.join(timeout=1.0)
            self.listener_thread = None
        if self.recv_sock:
            self.recv_sock.close()
            self.recv_sock = None
        self.logger_ri.info("Listener thread stopped.")

    def _listener_loop(self):
        """
        UDP listener loop for robot feedback messages.
        """
        self.logger_ri.info("Starting listener thread.")
        try:
            self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Bind to all interfaces on the specified port to receive data from robot
            bind_address = ("", self.robot_recv_address[1])  # Empty string means all interfaces
            self.recv_sock.bind(bind_address)
            self.recv_sock.settimeout(0.5)
            self.logger_ri.info(f"Listener thread started. Listening on port {self.robot_recv_address[1]} for robot messages...")
            self.logger_ri.info(f"Socket bound to: {self.recv_sock.getsockname()}")
            
            # Add a counter to track timeout cycles for debugging
            timeout_counter = 0
            
            while self.listener_running:
                try:
                    data, addr = self.recv_sock.recvfrom(4096)
                    if data:
                        self.logger_ri.info(f"Received {len(data)} bytes from {addr}: {data.decode(errors='ignore')[:200]}...")
                        # Reset timeout counter when we receive data
                        timeout_counter = 0
                        # Process robot feedback here if needed
                except socket.timeout:
                    # Normal - no data arrived within 0.5s
                    timeout_counter += 1
                    # Log every 60 timeouts (30 seconds) to show we're still listening
                    if timeout_counter % 60 == 0:
                        self.logger_ri.debug(f"Listener still active, waiting for data... ({timeout_counter * 0.5:.1f}s elapsed)")
                except Exception as e:
                    self.logger_ri.error(f"Error in listener thread: {e}")
                    break
                    
        except Exception as e:
            self.logger_ri.error(f"Error setting up listener thread: {e}")
        finally:
            self.logger_ri.info("Listener thread ended.")

    ################################################################
    # 6) DATA STORAGE
    ################################################################

    def _store_joint_data(self, robot_position, teachbot_position):
        """
        Store joint data in shared memory for recording.
        Gripper state should already be included as the last joint in position arrays.
        """
        robot_pos = robot_position.tolist() if isinstance(robot_position, np.ndarray) else robot_position
        teachbot_pos = teachbot_position.tolist() if isinstance(teachbot_position, np.ndarray) else teachbot_position
        
        joint_data = {
            "robot_position": robot_pos,
            "teachbot_position": teachbot_pos,
            "sent_robot_position": teachbot_pos,  # Use teachbot as sent position for TechMan
            "robot_position_timestamp": time.time(),
            "seq_id": getattr(self, 'seq_id', 0),
        }
        try:
            if self.shm_joint_data1.full():
                self.shm_joint_data1.get_nowait()
            self.shm_joint_data1.put(joint_data)
        except Exception as e:
            self.logger_ri.error(f"Could not store joint data: {e}")

    def _store_joint_data_with_ee(self, robot_position, teachbot_position):
        """
        Store joint data in shared memory for recording, including end-effector states.
        """
        robot_pos = robot_position.tolist() if isinstance(robot_position, np.ndarray) else robot_position
        teachbot_pos = teachbot_position.tolist() if isinstance(teachbot_position, np.ndarray) else teachbot_position
        
        # Ensure we have the full position including EE states
        if len(robot_pos) < self.total_dof:
            # Extend robot position with current EE states
            robot_pos_with_ee = robot_pos[:self.dof] + self.ee_dof_states
        else:
            robot_pos_with_ee = robot_pos
            
        if len(teachbot_pos) < self.total_dof:
            # Extend teachbot position with current EE states  
            teachbot_pos_with_ee = teachbot_pos[:self.dof] + self.ee_dof_states
        else:
            teachbot_pos_with_ee = teachbot_pos
            
        # Create joint data structure with end-effector information
        joint_data = {
            "robot_position": robot_pos_with_ee,
            "teachbot_position": teachbot_pos_with_ee,
            "sent_robot_position": robot_pos_with_ee,
            "robot_position_timestamp": time.time(),
            "seq_id": getattr(self, 'seq_id', 0),
            "gripper_state": self.gripper_state,
            "ee_dof_states": self.ee_dof_states.copy(),
        }
        try:
            if self.shm_joint_data1.full():
                self.shm_joint_data1.get_nowait()
            self.shm_joint_data1.put(joint_data)
        except Exception as e:
            self.logger_ri.error(f"Could not store joint data with EE: {e}")

    def extract_ee_dof_values(self, action_master):
        """
        Extract end effector DOF values from the action.
        Expects action_master to have self.total_dof elements (robot joints + EE DOFs).
        
        For external EE via MQTT:
        - eedof1: Potentiometer value (0.0-1.0) -> will be scaled to 0-4095 in MQTT send
        - eedof2: Grind value (0.0 or 1.0) -> will be converted to boolean in MQTT send
        
        Returns the EE DOF values as a list.
        """
        if len(action_master) >= self.total_dof:
            ee_dof_values = action_master[self.dof:self.dof + self.dof_ee]
            
            # Ensure eedof1 (setpoint) is within 0-1 range for scaling
            if self.ee_type == "external" and len(ee_dof_values) > 0:
                # Clamp eedof1 to 0-1 range for proper scaling to 12-bit word (0-4095)
                ee_dof_values[0] = max(0.0, min(1.0, ee_dof_values[0]))
                
                # Ensure eedof2 (grind) is 0 or 1
                if len(ee_dof_values) > 1:
                    ee_dof_values[1] = 1.0 if ee_dof_values[1] > 0.5 else 0.0
            
            return ee_dof_values
        else:
            # Fallback: use zeros if not enough elements
            self.logger_ri.warning(f"Action has {len(action_master)} elements, expected {self.total_dof}. Using zero EE DOF values.")
            return [0.0] * self.dof_ee

    def determine_gripper_state(self, gripper_state):
        """
        Decide when to activate / deactivate the gripper based on threshold and
        internal 'gripper_delay' logic. Sets gripper_on/gripper_off flags for logging.
        """
        now = time.time()
        self.gripper_on = False
        self.gripper_off = False
        
        # Turn ON
        if gripper_state >= self.gripper_treshold and not self.gripper_state:
            if self.gripper_delay > 0.0:
                self.gripper_state_change_time_threshold = self.gripper_delay / getattr(self, 'robot_speed', 1.0)
                # Check if enough time has passed since last state change
                if (now - self.gripper_state_change_time) >= self.gripper_state_change_time_threshold:
                    self.gripper_on = True
                    self.gripper_state = 1
                    self.gripper_state_change_time = now
                    self.logger_ri.info(f"Gripper turned ON: state={gripper_state:.3f}, threshold={self.gripper_treshold}")
            else:
                self.gripper_on = True
                self.gripper_state = 1
                self.gripper_state_change_time = now
                self.logger_ri.info(f"Gripper turned ON (no delay): state={gripper_state:.3f}, threshold={self.gripper_treshold}")

        # Turn OFF
        elif gripper_state < self.gripper_treshold and self.gripper_state:
            self.gripper_off = True
            self.gripper_state = 0
            self.gripper_state_change_time = now
            self.logger_ri.info(f"Gripper turned OFF: state={gripper_state:.3f}, threshold={self.gripper_treshold}")

    ################################################################
    # 7) ADDITIONAL ROBOT METHODS (for compatibility)
    ################################################################

    def push_joint_motion(self, joint_position, velocity_scale=1.0, wait=True):
        """
        Push a joint motion command to the robot.
        """
        if not self.connected:
            self.logger_ri.error("Robot not connected")
            return False

        try:
            position = action_master_TM_translation(joint_position, self.logger_ri)
            script_cmd = "Position({:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f})".format(*position)
            return send_tmsct_cmd(self.sock, "1", script_cmd, self.logger_ri)
        except Exception as e:
            self.logger_ri.error(f"Error pushing joint motion: {e}")
            return False

    def get_joint_position(self):
        """
        Get current joint position. Returns start position as Techman doesn't provide real-time feedback.
        """
        # Return last known position or start position
        return self.start_position

    def get_joint_velocity(self):
        """
        Get current joint velocity. Returns zeros as Techman doesn't provide real-time velocity feedback.
        """
        return [0.0] * self.dof

    def set_gripper_state(self, state):
        """
        Set gripper state with logging and EE DOF state synchronization.
        """
        if state != self.gripper_state:
            old_state = self.gripper_state
            self.gripper_state = state
            self.gripper_state_change_time = time.time()
            self.logger_ri.info(f"Gripper state changed from {old_state} to {state}")
            
            # Update the first EE DOF state to match gripper state
            if self.dof_ee > 0:
                self.ee_dof_states[0] = float(state)
                self.logger_ri.info(f"Updated EE DOF states: {self.ee_dof_states}")
        return True

    def get_ee_dof_states(self):
        """
        Get current end-effector DOF states.
        """
        return self.ee_dof_states.copy()

    def test_udp_reception(self, test_duration=10):
        """
        Test UDP reception for a specified duration to diagnose connectivity issues.
        This can be called manually to check if any UDP traffic is arriving.
        """
        self.logger_ri.info(f"Starting UDP reception test for {test_duration} seconds...")
        
        try:
            # Create a test socket
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to the same port as the listener
            bind_address = ("", self.robot_recv_address[1])
            test_sock.bind(bind_address)
            test_sock.settimeout(1.0)
            
            self.logger_ri.info(f"Test socket bound to: {test_sock.getsockname()}")
            self.logger_ri.info(f"Listening for any UDP traffic on port {self.robot_recv_address[1]}...")
            
            packets_received = 0
            start_time = time.time()
            
            while (time.time() - start_time) < test_duration:
                try:
                    data, addr = test_sock.recvfrom(4096)
                    packets_received += 1
                    self.logger_ri.info(f"UDP packet #{packets_received} from {addr}: {len(data)} bytes")
                    self.logger_ri.info(f"Data preview: {data[:100] if len(data) > 100 else data}")
                    
                except socket.timeout:
                    # No data within 1 second
                    pass
                    
            test_sock.close()
            
            if packets_received == 0:
                self.logger_ri.warning(f"No UDP packets received during {test_duration} second test")
                self.logger_ri.info("Possible issues:")
                self.logger_ri.info("1. Robot not configured to send data to this IP/port")
                self.logger_ri.info("2. Firewall blocking incoming UDP on port 5891")
                self.logger_ri.info("3. Robot not in correct mode for data transmission")
                self.logger_ri.info("4. Network routing issues")
            else:
                self.logger_ri.info(f"Test completed: {packets_received} UDP packets received")
                
        except Exception as e:
            self.logger_ri.error(f"Error during UDP test: {e}")

    def check_robot_feedback_config(self):
        """
        Send commands to check/configure robot feedback settings.
        This tries to enable data transmission from the robot.
        """
        if not self.connected:
            self.logger_ri.error("Cannot configure robot feedback - not connected")
            return False
            
        self.logger_ri.info("Attempting to configure robot feedback...")
        
        try:
            # Try to enable data transmission to our IP and port
            local_ip = self.robot_recv_address[0] if self.robot_recv_address[0] != "" else "192.168.10.3"
            local_port = self.robot_recv_address[1]
            
            # Example commands to enable feedback (these may need adjustment for your specific robot)
            feedback_commands = [
                f'ScriptExit()',  # Ensure we're not in a script
                f'SetTCPSpeedLimit(false)',  # Disable speed limits for feedback
                # Add specific Techman commands to enable UDP feedback if available
                # f'SendData("{local_ip}", {local_port}, "enable")',  # Example - adjust based on robot manual
            ]
            
            for cmd in feedback_commands:
                self.logger_ri.info(f"Sending feedback config command: {cmd}")
                if not send_tmsct_cmd(self.sock, "1", cmd, self.logger_ri):
                    self.logger_ri.warning(f"Failed to send command: {cmd}")
                time.sleep(0.1)  # Small delay between commands
                
            self.logger_ri.info("Robot feedback configuration commands sent")
            return True
            
        except Exception as e:
            self.logger_ri.error(f"Error configuring robot feedback: {e}")
            return False


##################################################################
# MAIN FUNCTION
##################################################################
def run_robot_interface(robot_interface_commup, robot_interface_commdown, shm_target_pos1, shm_target_pos2_info, shm_joint_data1, shm_joint_data2, setup_id=None):
    """
    Main function to run the TechmanRobot interface in a loop,
    checking for commands on 'robot_interface_commdown' queue.
    """
    def get_base_interface_name(interface_name):
        """Extract base interface name from setup-specific interface name."""
        if setup_id and interface_name.startswith(f"{int(setup_id):02d}_"):
            underscore_pos = interface_name.find('_')
            if underscore_pos != -1:
                return interface_name[underscore_pos + 1:]
        return interface_name
    
    if setup_id:
        component_tag = f"{int(setup_id):02d}_ROBOT_INTERFACE"
    else:
        component_tag = "ROBOT_INTERFACE"
    logger_ri = setup_logging(component_tag)

    logger_ri.info("Starting Robot Interface...")

    # Load the config
    config = load_config()
    
    # Use setup-specific config if setup_id is provided
    if setup_id is not None:
        setup_name = f"setup_{setup_id}"
        logger_ri.info(f"Using setup-specific config for setup: {setup_name}")
        if "hardware" in config and setup_name in config["hardware"]:
            # Create effective config with setup-specific hardware
            hw_config = config["hardware"][setup_name]
            
            # Merge with global hardware config as fallback
            effective_hw_config = {**config.get("hardware", {}), **hw_config}
            
            # Update config to use setup-specific hardware
            config = {**config}  # Shallow copy
            config["hardware"] = effective_hw_config
        else:
            logger_ri.warning(f"Setup {setup_name} not found in config, using global hardware config")
    
    logger_ri.info("Config loaded successfully.")

    # Robot brand name (e.g., 'Techman'), capitalized
    robot_brand = config["hardware"]["robot"]["brand"].lower().capitalize()
    check_queue_period = config["general"]["check_queue_period"]
    robot = None

    logger_ri.info(f"Starting {robot_brand} Robot Interface")

    # Dynamically instantiate the correct robot class
    try:
        robot_class_name = f"{robot_brand}Robot"
        # Use direct class reference instead of globals() lookup
        if robot_class_name == "TechmanRobot":
            robotClass = TechmanRobot
        else:
            # Fallback to globals() for other robot types
            robotClass = globals()[robot_class_name]
        robot = robotClass(robot_interface_commup, shm_target_pos1, shm_target_pos2_info, shm_joint_data1, shm_joint_data2, logger_ri, config)
        logger_ri.info("Initialized %s Robot Interface", robot_brand)
        # Send a success response for "initialization"
        send_response(
            logger_ri,
            robot_interface_commup,
            {"interface": component_tag, "message": "initialization"},
            error="None",
            setup_id=setup_id
        )
    except Exception as e:
        logger_ri.error(
            "Error initializing %s class: %s", robot_class_name, e
        )
        # Send an error response
        send_response(
            logger_ri,
            robot_interface_commup,
            {"message": "initialization", "interface": component_tag},
            error=f"{e}",
            setup_id=setup_id
        )

    # Listen for commands in a loop
    while True:
        if not robot_interface_commdown.empty():
            full_message = robot_interface_commdown.get()
            logger_ri.info(f"Message received: {full_message}")

            msg_type = full_message.get("type", "")
            msg_interface = full_message.get("interface", "")
            message = full_message.get("message", "")
            
            # Extract base interface name for comparison
            base_interface = get_base_interface_name(msg_interface)

            if msg_type == "CMD" and base_interface == "ROBOT_INTERFACE":
                if message == "stop":
                    logger_ri.info("[CMD] stop robot_interface and %s robot", robot_brand)
                    if robot is None:
                        pass
                    elif robot.connected:
                        robot.stop()
                        logger_ri.info("Stop from %s robot", robot_brand)
                    else:
                        logger_ri.warning("stop unsuccessful, robot is not connected from %s robot", robot_brand)

                    send_response(logger_ri, robot_interface_commup, full_message, error="None")
                    break

                elif message == "start_teleoperation":
                    if robot.robot_running:
                        logger_ri.warning(
                            "[CMD] start_teleoperation unsuccessful, Robot already running"
                        )
                        send_response(logger_ri, robot_interface_commup, full_message,
                                      error="Robot already running")
                    else:
                        if robot.start_teleoperation(full_message):
                            logger_ri.info("start_teleoperation successful")
                            send_response(logger_ri, robot_interface_commup, full_message, error="None")
                        else:
                            logger_ri.error("start_teleoperation unsuccessful, Could not start teleop")
                            send_response(logger_ri, robot_interface_commup, full_message,
                                          error="Could not start teleop, try again")

                elif message == "start_teleoperation_record":
                    if robot.robot_running:
                        logger_ri.warning(
                            "[CMD] start_teleoperation_record unsuccessful, Robot already running"
                        )
                        send_response(logger_ri, robot_interface_commup, full_message,
                                      error="Robot already running")
                    else:
                        if robot.start_teleoperation_record(full_message):
                            logger_ri.info("start_teleoperation_record successful")
                            send_response(logger_ri, robot_interface_commup, full_message, error="None")
                        else:
                            logger_ri.error(
                                "start_teleoperation_record unsuccessful, Could not start teleop."
                            )
                            send_response(logger_ri, robot_interface_commup, full_message,
                                          error="Could not start teleop, try again")
                            
                elif message == "record_episodes":
                    if robot.robot_running:
                        logger_ri.warning(
                            "[CMD] record_episodes unsuccessful, Robot already running"
                        )
                        send_response(logger_ri, robot_interface_commup, full_message,
                                      error="Robot already running")
                    else:
                        if robot.record_episodes(full_message):
                            logger_ri.info("record_episodes successful")
                            send_response(logger_ri, robot_interface_commup, full_message, error="None")
                        else:
                            logger_ri.error("record_episodes unsuccessful, Could not record episodes")
                            send_response(logger_ri, robot_interface_commup, full_message,
                                          error="Could not record episodes")

                elif message == "play_recording":
                    if robot.robot_running:
                        logger_ri.warning(
                            "[CMD] play_recording unsuccessful, Robot already running"
                        )
                        send_response(logger_ri, robot_interface_commup, full_message,
                                      error="Robot already running")
                    else:
                        if robot.play_recording(full_message):
                            logger_ri.info("play_recording successful")
                            send_response(logger_ri, robot_interface_commup, full_message, error="None")
                        else:
                            logger_ri.error("play_recording unsuccessful, Could not play recording")
                            send_response(logger_ri, robot_interface_commup, full_message,
                                          error="Could not play recording")

                elif message == "run_policy":
                    if robot.robot_running:
                        logger_ri.warning(
                            "[CMD] run_policy unsuccessful, Robot already running"
                        )
                        send_response(logger_ri, robot_interface_commup, full_message,
                                      error="Robot already running")
                    else:
                        if robot.run_policy(full_message):
                            logger_ri.info("run_policy successful")
                            send_response(logger_ri, robot_interface_commup, full_message, error="None")
                        else:
                            logger_ri.error("run_policy unsuccessful, Could not run policy")
                            send_response(logger_ri, robot_interface_commup, full_message,
                                          error="Could not run policy")
                            
                else:
                    logger_ri.error("Unknown dict CMD message: %s", full_message)
                    send_response(logger_ri, robot_interface_commup, full_message,
                                  error="Unknown CMD message")
            else:
                logger_ri.error("Unknown dict message format: %s", full_message)
                send_response(logger_ri, robot_interface_commup, full_message,
                              error="Unknown dict message format")

        time.sleep(check_queue_period)
