import time
import numpy as np
import threading
import os
import sys
import socket
from threading import Thread
from collections import deque
from .teachbot_0interface import TeachbotInterface

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.utils import setup_logging, load_config


#################################################################
# Helper: Same response function as in dummy code
#################################################################

def send_response(logger_ti, teachbot_interface_commup, payload, error="None", **kwargs):
    """
    Helper to construct and send a consistent response dict:
        {
         "type": "RESP",
         "message": <whatever command was processed>,
         "error": <error string or "None">,
         ... plus optional extra fields ...
        }
    Then puts that response into the `teachbot_interface_commup` queue.
    """
    response = payload.copy()
    response["type"] = "RESP"
    if error not in ("None", ""):
        response["error"] = error
    elif response.get("error", "") == "":
        response["error"] = "None"

    # Merge additional kwargs
    response.update(kwargs)

    # Log and publish
    logger_ti.info(f"Preparing to send response: {response}")
    teachbot_interface_commup.put(response)
    logger_ti.info(f"Sent response: {response}")



#################################################################
# The TosTeachbot class
#################################################################

class TosTeachbot(TeachbotInterface):

    def __init__(self, teachbot_interface_commup, teachbot_interface_commdown, shm_target_pos1, logger_ti, config):
        self.teachbot_interface_commup = teachbot_interface_commup
        self.teachbot_interface_commdown = teachbot_interface_commdown
        self.shm_target_pos1 = shm_target_pos1
        self.logger_ti = logger_ti
        self.config = config        # Config variables
        try:
            self.control_loop_language = config["general"]["control_loop_language"]
            self.status_refresh_period = config["general"]["status_refresh_period"]
            self.target_pos_period = config["hardware"]["robot"]["control_dt"]
            self.min_joint6 = config["hardware"]["teachbot"]["j6_min_angle"]
            self.max_joint6 = config["hardware"]["teachbot"]["j6_max_angle"]
            self.joint6_multiplier = config["hardware"]["teachbot"]["j6_multiplier"]
            self.joint4_locked = config["hardware"]["teachbot"]["j4_locked"]
            self.teachbot_gripper_treshold = config["hardware"]["teachbot"].get("gripper_treshold", 1085)
        except KeyError as e:
            self.logger_ti.error(f"Missing required config key: {e}")
            raise

            # TOS encoder configuration from config
        tos_config = self.config.get("hardware", {}).get("teachbot", {})
        self.local_ip = tos_config.get("local_ip", "192.168.1.201")
        self.start_port = tos_config.get("start_port", 5004)
        self.num_joints = tos_config.get("num_joints", 6)
        self.packet_size = tos_config.get("packet_size", 3)
          # Port to joint mapping - allows flexible assignment of ports to joints
        # Format: {joint_index: port_offset} where port = start_port + port_offset
        raw_mapping = tos_config.get("port_joint_mapping", {
            0: 0,  # Joint 0 -> port 5004 (start_port + 0)
            1: 1,  # Joint 1 -> port 5005 (start_port + 1)
            2: 2,  # Joint 2 -> port 5006 (start_port + 2)
            3: 3,  # Joint 3 -> port 5007 (start_port + 3)
            4: 4,  # Joint 4 -> port 5008 (start_port + 4)
            5: 5   # Joint 5 -> port 5009 (start_port + 5)
        })
        
        # Convert string keys to integers if needed (YAML might load as strings)
        self.port_joint_mapping = {}
        for k, v in raw_mapping.items():
            self.port_joint_mapping[int(k)] = int(v)# Position offset for zero position calibration
        self.position_offsets = tos_config.get("position_offsets", [0] * self.num_joints)
        self.joint_scale_factors = tos_config.get("joint_scale_factors", [1.0] * self.num_joints)

        # Streams
        self.joint_target_streaming = False
        self.joint_target_streaming_thread = None

        self.logger_ti.info("Initialized TOS Teachbot, now connecting to encoders...")
        self.connect()
        self.start_joint_target_streaming()


    ##########################################################
    # Commands
    ##########################################################    
    
    def stop(self):
        # Stop streaming first
        if self.joint_target_streaming:
            self.stop_joint_target_streaming()        # Stop and cleanup encoder threads
        if hasattr(self, 'encoder_threads') and self.encoder_threads:
            self.logger_ti.info("Stopping TOS encoder threads...")
            for joint_idx, thread in self.encoder_threads.items():
                thread.stop()
                thread.join(timeout=1.0)
                self.logger_ti.info(f"Stopped encoder thread for joint {joint_idx}")
            self.encoder_threads = {}

        self.connected = False
        self.logger_ti.info("TOS Teachbot stopped successfully")


    ###########################################################
    # Connect to the robot
    ###########################################################

    def connect(self):
        """
        Initialize the TOS encoder listeners and set up the connection.
        """
        try:
            self.logger_ti.info("Connecting to TOS encoders...")

              # Create encoder listeners for each joint using the port mapping
            self.encoder_threads = {}  # Change to dict to map joint_index -> thread
            self.joint_to_port = {}    # Map joint index to actual port number
            self.logger_ti.info(f"Using local IP: {self.local_ip}, start port: {self.start_port}, packet size: {self.packet_size}")
            for joint_index in range(self.num_joints):
                if joint_index in self.port_joint_mapping:
                    port_offset = self.port_joint_mapping[joint_index]
                    port = self.start_port + port_offset
                    self.joint_to_port[joint_index] = port
                    
                    thread = ListenerThread(self.local_ip, port, packet_size=self.packet_size)
                    thread.start()
                    self.encoder_threads[joint_index] = thread
                    self.logger_ti.info(f"Started encoder listener for joint {joint_index} on port {port}")
                else:
                    self.logger_ti.warning(f"No port mapping found for joint {joint_index}")
            
            self.logger_ti.info(f"Joint to port mapping: {self.joint_to_port}")
            
            # Wait a moment for connections to establish
            time.sleep(0.5)
            
            self.connected = True
            self.logger_ti.info("Successfully connected to TOS encoders")
            return True
            
        except Exception as e:
            self.logger_ti.error(f"Failed to connect to TOS encoders: {e}")
            self.connected = False
            return False


    ##########################################################
    # Joint Target Streaming
    ##########################################################

    def start_joint_target_streaming(self):
        if not self.connected:
            return False

        self.joint_target_streaming = True
        self.joint_target_streaming_thread = Thread(target=self.joint_target_updating)
        self.joint_target_streaming_thread.start()
        return True

    def stop_joint_target_streaming(self):
        self.joint_target_streaming = False
        if self.joint_target_streaming_thread:
            self.joint_target_streaming_thread.join(0.5)
        self.joint_target_streaming_thread = None
        return True

    def joint_target_updating(self):
        """
        Periodically read the robot’s actual joint states,
        translate them, and put them in shm_target_pos1.
        """
        self.logger_ti.info("Starting joint target updating thread...")
        while self.joint_target_streaming:
            if self.connected and hasattr(self, 'encoder_threads') and self.encoder_threads:
                try:                    # Read raw encoder positions from all joints in order
                    joint_states = [0.0] * self.num_joints  # Initialize with zeros
                    
                    for joint_index in range(self.num_joints):
                        if joint_index in self.encoder_threads:
                            encoder_thread = self.encoder_threads[joint_index]
                            encoder_data = encoder_thread.get_data()
                            raw_position = encoder_data["position"]

                            # Check for encoder errors
                            if encoder_data["error_bit"]:
                                self.logger_ti.warning(f"Encoder error on joint {joint_index} (port {self.joint_to_port.get(joint_index, 'unknown')})")
                            if encoder_data["warning_bit"]:
                                self.logger_ti.warning(f"Encoder warning on joint {joint_index} (port {self.joint_to_port.get(joint_index, 'unknown')})")
                            
                            joint_states[joint_index] = raw_position
                            # self.logger_ti.info(f"Joint {joint_index} (port {self.joint_to_port.get(joint_index)}): {raw_position}")
                        else:
                            self.logger_ti.warning(f"No encoder thread for joint {joint_index}")
                            joint_states[joint_index] = 0.0  # Default to 0 if no encoder
                    
                    # Add gripper state (placeholder for now)
                    if len(joint_states) == self.num_joints:
                        joint_states.append(0.0)  # Gripper position placeholder
                    joint_states_translated, safe = self.action_translation_and_check(joint_states)

                    # If safe, push to shared memory
                    if safe:
                        if self.control_loop_language == "python":
                            if self.shm_target_pos1.full():
                                self.shm_target_pos1.get_nowait()
                            self.shm_target_pos1.put(joint_states_translated, timeout=0.1)
                    else:
                        self.logger_ti.warning("Joint target not safe: %s", joint_states_translated)

                except Exception as e:
                    self.logger_ti.error("Error updating joint target: %s", e)

            time.sleep(self.target_pos_period)

    def action_translation_and_check(self, action):
        """
        Translate TOS encoder positions to robot joint angles and check for safety.
        Set offset to set the 0 degree positions right with the robot.
        Returns the translated action and whether it is safe.
        """
        try:
            if len(action) < self.num_joints:
                self.logger_ti.warning(f"Insufficient joint data: got {len(action)}, expected {self.num_joints}")
                return action, False
            
            translated_action = []
            
            # Convert encoder positions to degrees for each joint
            for i in range(self.num_joints):
                raw_position = action[i]
                
                # Convert from encoder counts to degrees
                # For 17-bit encoder: 2^17 = 131072 counts per revolution
                max_counts = 1 << 17  # 131072 for 17-bit encoder
                degrees_per_count = 360.0 / max_counts
                
                # Apply position offset for zero calibration
                offset = self.position_offsets[i] if i < len(self.position_offsets) else 0
                position_with_offset = raw_position - offset
                
                # Convert to degrees
                degrees = (position_with_offset * degrees_per_count) % 360.0
                  # Convert to -180 to +180 range
                if degrees > 180.0:
                    degrees -= 360.0
                
                # Apply joint-specific scaling factors
                scale_factor = self.joint_scale_factors[i] if i < len(self.joint_scale_factors) else 1.0
                degrees *= scale_factor
                
                # Apply joint6 clamping like in interbotix teachbot
                if i == 5:  # Joint 6 (0-based index)
                    degrees = max(self.min_joint6, min(degrees, self.max_joint6))
                
                translated_action.append(degrees)
                
                # self.logger_ti.info(f"Joint {i+1}: raw={raw_position}, offset={offset}, degrees={degrees:.2f}")
            
            # Add gripper if present
            if len(action) > self.num_joints:
                translated_action.append(action[self.num_joints])  # Pass through gripper value
            
            # Safety checks
            safe = True
            # safe = self.check_joint_limits(translated_action)
            
            return translated_action, safe
            
        except Exception as e:
            self.logger_ti.error(f"Error in action translation: {e}")
            return action, False
    
    def check_joint_limits(self, joint_positions):
        """
        Check if joint positions are within safe limits.
        Returns True if safe, False otherwise.
        """
        try:
            # Get joint limits from config
            robot_config = self.config.get("hardware", {}).get("robot", {})
            upper_limits = robot_config.get("upper_limits", [165, 140, 208, 185, 120, 355])
            lower_limits = robot_config.get("lower_limits", [-165, -95, -65, -185, -120, -355])
            
            for i, pos in enumerate(joint_positions[:6]):  # Check first 6 joints
                if i < len(upper_limits) and i < len(lower_limits):
                    if pos > upper_limits[i] or pos < lower_limits[i]:
                        # self.logger_ti.warning(f"Joint {i+1} out of limits: {pos} (limits: {lower_limits[i]} to {upper_limits[i]})")
                        return False
            
            return True
            
        except Exception as e:
            self.logger_ti.error(f"Error checking joint limits: {e}")
            return False




#################################################################
# The main interface loop
#################################################################

def run_teachbot_interface(teachbot_interface_commup, teachbot_interface_commdown, shm_target_pos1):
    # Load the config
    config = load_config()

    # Setup logging
    component_tag = "TEACHBOT_INTERFACE"
    logger_ti = setup_logging(component_tag)
    
    robot_brand = config["hardware"]["teachbot"]["brand"].lower().capitalize()
    check_queue_period = config["general"]["check_queue_period"]

    logger_ti.info("Starting %s Teachbot Interface", robot_brand)

    teachbot = None
    teachbot_class_name = f"{robot_brand}Teachbot"
    try:
        TeachbotClass = globals()[teachbot_class_name]
    except KeyError:
        logger_ti.error("Teachbot class %s not found in this module.", teachbot_class_name)
        # Send an error response and return
        send_response(
            logger_ti,
            teachbot_interface_commup,
            {"interface": component_tag, "message": "initialization"},
            error=f"Teachbot class {teachbot_class_name} not found."
        )
        return

    # Instantiate the teachbot
    try:
        teachbot = TeachbotClass(
            teachbot_interface_commup,
            teachbot_interface_commdown,
            shm_target_pos1,
            logger_ti,
            config
        )
        logger_ti.info("Successfully initialized %s class", teachbot_class_name)

        # Respond that we are up and running
        send_response(
            logger_ti,
            teachbot_interface_commup,
            {"interface": component_tag, "message": "initialization"},
            error="None"
        )
    except Exception as e:
        logger_ti.error("Error instantiating %s: %s", teachbot_class_name, e)
        send_response(
            logger_ti,
            teachbot_interface_commup,
            {"interface": component_tag, "message": "initialization"},
            error=str(e)
        )
        return

    # Main loop to read commands
    while True:
        if not teachbot_interface_commdown.empty():
            full_message = teachbot_interface_commdown.get()
            logger_ti.info("Received message: %s", full_message)

            msg_type = full_message.get("type", "")
            msg_interface = full_message.get("interface", "")
            message = full_message.get("message", "")

            if msg_type == "CMD" and msg_interface == "TEACHBOT_INTERFACE":
                if message == "stop":
                    try:
                        if teachbot.connected:
                            teachbot.stop()
                            logger_ti.info("stop from %s Teachbot", robot_brand)
                        else:
                            logger_ti.info("Already stop from %s Teachbot", robot_brand)
                        send_response(logger_ti, teachbot_interface_commup, full_message, error="None")
                    except Exception as ex_disc:
                        logger_ti.error("Error stop from Teachbot: %s", ex_disc)
                        send_response(logger_ti, teachbot_interface_commup, full_message, error=str(ex_disc))
                    break

                else:
                    logger_ti.warning("Unknown command: %s", full_message)
                    send_response(logger_ti, teachbot_interface_commup, full_message, error="Unknown command")
            else:
                logger_ti.warning("Unknown message: %s", full_message)
                send_response(logger_ti, teachbot_interface_commup, full_message, error="Unknown message")

        time.sleep(check_queue_period)














###############################################################################
# 1) ListenerThread for each encoder port
###############################################################################
class ListenerThread(threading.Thread):
    def __init__(self, local_ip, local_port, packet_size=3, max_timestamps=100):
        """
        Creates a UDP socket listening at (local_ip, local_port).
        Reads frames of size 'packet_size' from a 17-bit singleturn encoder.
        If you need 6-byte frames ('d'), set packet_size=6, etc.
        """
        super().__init__()
        self.local_ip = local_ip
        self.local_port = local_port
        self.packet_size = packet_size
        self.max_timestamps = max_timestamps

        # Create the socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)
        self.sock.bind((local_ip, local_port))

        self.running = False

        # Shared data
        self.lock = threading.Lock()
        self.position = 0
        self.error_bit = False
        self.warning_bit = False
        self.detailed_status = 0

        # Timestamps for frequency
        self.timestamps = deque(maxlen=max_timestamps)
        self.latest_freq = 0.0
        self.last_time = None

    def run(self):
        self.running = True
        leftover = b''

        while self.running:
            try:
                # Read exactly 'packet_size' bytes (or partial chunk)
                data = self.sock.recv(self.packet_size)
            except socket.timeout:
                continue
            except OSError:
                break

            if not data:
                continue

            leftover += data

            # Extract each frame
            while len(leftover) >= self.packet_size:
                frame = leftover[:self.packet_size]
                leftover = leftover[self.packet_size:]
                self.parse_frame(frame)

    def parse_frame(self, frame: bytes):
        now = time.perf_counter()

        with self.lock:
            if self.packet_size == 3:
                # 17-bit singleturn: 3 bytes
                # bits 1..0 => E/W, bits 6..2 => zero padding, bits 23..7 => position
                raw_val = (frame[0] << 16) | (frame[1] << 8) | frame[2]
                w_bit = raw_val & 0x01
                e_bit = (raw_val >> 1) & 0x01
                pos = raw_val >> 7
                self.position = pos
                self.error_bit = (e_bit == 0)
                self.warning_bit = (w_bit == 0)
                self.detailed_status = 0

            elif self.packet_size == 4:
                # e.g. command '1'
                header = frame[0]
                pos_bytes = frame[1:4]
                raw_val = (pos_bytes[0] << 16) | (pos_bytes[1] << 8) | pos_bytes[2]
                w_bit = raw_val & 0x01
                e_bit = (raw_val >> 1) & 0x01
                pos = raw_val >> 7
                self.position = pos
                self.error_bit = (e_bit == 0)
                self.warning_bit = (w_bit == 0)
                self.detailed_status = 0

            elif self.packet_size == 6:
                # e.g. command 'd'
                header = frame[0]
                pos_bytes = frame[1:4]
                stat_bytes = frame[4:6]
                raw_val = (pos_bytes[0] << 16) | (pos_bytes[1] << 8) | pos_bytes[2]
                w_bit = raw_val & 0x01
                e_bit = (raw_val >> 1) & 0x01
                pos = raw_val >> 7
                ds = (stat_bytes[0] << 8) | stat_bytes[1]
                self.position = pos
                self.error_bit = (e_bit == 0)
                self.warning_bit = (w_bit == 0)
                self.detailed_status = ds

            # Frequency
            self.timestamps.append(now)
            if len(self.timestamps) >= 2:
                dt_total = self.timestamps[-1] - self.timestamps[0]
                if dt_total > 0:
                    self.latest_freq = (len(self.timestamps) - 1) / dt_total
                else:
                    self.latest_freq = 0.0

            if self.last_time is not None:
                dt = now - self.last_time
                # print(f"Port {self.local_port}: Δt={dt:.6f}s, pos={self.position}")
            self.last_time = now

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except:
            pass

    def get_data(self):
        with self.lock:
            return {
                "position": self.position,
                "error_bit": self.error_bit,
                "warning_bit": self.warning_bit,
                "detailed_status": self.detailed_status,
                "freq": self.latest_freq
            }


