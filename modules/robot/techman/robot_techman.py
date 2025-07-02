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


def action_master_TM_translation(master_pos_deg):
    """
    Transform master position to Techman robot coordinates.
    This function applies the necessary transformations for the Techman robot.
    """
    action = np.array(master_pos_deg, dtype=float)
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


def apply_joint_speed_limits(current_position, previous_position, previous_timestamp, speed_limits, dt, logger):
    """
    Apply speed limits to joint positions to ensure safe operation.
    
    Args:
        current_position: Target joint position (degrees)
        previous_position: Previous joint position (degrees) 
        previous_timestamp: Timestamp of previous position
        speed_limits: List of speed limits for each joint (degrees/second)
        dt: Control timestep (seconds)
        logger: Logger instance
        
    Returns:
        Tuple of (limited_position, new_timestamp)
    """
    current_time = time.time()
    
    # If no previous position, return current position unchanged
    if previous_position is None or previous_timestamp is None:
        return current_position, current_time
    
    # Calculate actual time delta
    actual_dt = current_time - previous_timestamp
    if actual_dt <= 0:
        actual_dt = dt  # Fallback to control timestep
    
    limited_position = []
    
    for i in range(len(current_position)):
        if i < len(previous_position) and i < len(speed_limits):
            # Calculate required velocity
            position_diff = current_position[i] - previous_position[i]
            required_velocity = abs(position_diff) / actual_dt
            
            # Check if velocity exceeds limit
            if required_velocity > speed_limits[i]:
                # Limit the position change
                max_change = speed_limits[i] * actual_dt
                if position_diff > 0:
                    limited_pos = previous_position[i] + max_change
                else:
                    limited_pos = previous_position[i] - max_change
                
                limited_position.append(limited_pos)
            else:
                limited_position.append(current_position[i])
        else:
            # No limit for this joint or invalid index
            limited_position.append(current_position[i])
    
    return limited_position, current_time


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
class TechmanRobot(RobotInterface):
    """
    Handles:
      - TMSCT connection (TCP) to a Techman robot,
      - UDP listening for status/feedback from robot,
      - Methods for motion commands (teleoperation, recording, playback),
      - Safety checks and limit monitoring,
      - Consistent structure matching FanucRobot implementation.
    """

    def __init__(self, robot_interface_commup, shm_target_pos1, shm_target_pos2_info, shm_joint_data1, shm_joint_data2, logger_ri, config):
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

        # Logging and config
        self.logger_ri = logger_ri
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
        self.default_recording_speed = config["general"]["default_recording_speed"]
        self.gripper_treshold = config["general"]["gripper_treshold"]
        self.gripper_delay = config["general"]["gripper_delay"]
        
        # Techman-specific config
        self.robot_address = tuple(config["hardware"]["robot"]["robot_adress"])
        self.robot_recv_address = tuple(config["hardware"]["robot"]["robot_recv_address"])
        
        # Use default limits if not specified in config
        self.upper_limits = config["hardware"]["robot"].get("upper_limits", [180, 180, 180, 180, 180, 180])
        self.lower_limits = config["hardware"]["robot"].get("lower_limits", [-180, -180, -180, -180, -180, -180])
        
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
        self.gripper_state = False
        self.gripper_state_change_time = 0
        self.master_positions = deque()

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

        # Speed limiter configuration and state
        self.joint_speed_limits = [120.0, 120.0, 180.0, 180.0, 180.0, 180.0]  # degrees per second
        self.previous_joint_position = None
        self.previous_timestamp = None
        self.speed_limiter_enabled = True

        # Connect automatically if desired:
        if not self.connect():
            raise Exception(
                "TechmanRobot: Could not connect to robot. Check IP address, connection, and robot state."
            )

    ################################################################
    # 1) COMMANDS
    ################################################################

    def start_teleoperation(self, full_message):
        """
        Move to the start position and start teleoperation.
        Create a thread to listen for target positions, then
        start the teleoperation control loop in another thread.
        """
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

    def play_recording(self, full_message=None):
        """
        Play the recorded positions from a JSON file.
        """
        if full_message is None:
            full_message = {"message": "play_recording"}
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
        self.logger_ri.info("stop, disconnected.")
        return success

    def disconnect(self):
        """
        Disconnect from the Techman robot.
        """
        if not self.connected:
            self.logger_ri.warning("Already disconnected.")
            return True

        # Close sockets
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
            filename = get_data_path(self.config, filename)
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

            # Load positions
            self.master_positions = deque([s["master_position"] for s in data["samples"]])
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
        Core loop for either teleoperation or playing back a recorded trajectory.
        """
        if self.play_recording_active:
            self._execute_play_recording()
        else:
            self._execute_teleoperation()

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
            send_tc_command(self.robot_interface_commup, {"type": "CMD", "message": "stop", "interface": "ROBOT_INTERFACE"})
            self.logger_ri.info("_send_stop_response, sent stop command to queue.")

    def _execute_teleoperation(self):
        """
        Execute teleoperation control loop.
        Main teleoperation loop - no opening ceremony here as it's handled in _check_start_position.
        """
        # Main teleoperation loop
        self.logger_ri.info("Starting teleoperation control loop.")
        while self.robot_running:
            start_time = time.time()

            if self.target_pos_received is not None:
                action_deg = self.target_pos_received

                # Safety check
                if not action_master_safety_check(action_deg, self.upper_limits, self.lower_limits):
                    self.logger_ri.warning("Safety check failed for master position.")
                    break

                # Send TMSCT command
                position = action_master_TM_translation(copy.deepcopy(action_deg))
                script_cmd = "Position({:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f})".format(*position)
                
                if not send_tmsct_cmd(self.sock, "2", script_cmd, self.logger_ri):
                    self.logger_ri.error("Error sending command to Techman.")
                    break
                self.logger_ri.info(f"Sent position command: {script_cmd}")

                # Store joint data if recording
                if self.recording:
                    robot_position = self.start_position  # Use actual position if available
                    self._store_joint_data(robot_position, action_deg)

            else:
                self.logger_ri.info("No target position received, skipping control step.")

            # Rate control
            elapsed = time.time() - start_time
            sleep_time = self.control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.stop_robot_running(script_cmd)

    def opening_ceremony(self):
        """
        Wait for target position to be close to start position before beginning teleoperation.
        This matches the Fanuc robot's opening ceremony logic.
        """
        self.logger_ri.info("Starting opening ceremony - waiting for target position near start position")
        
        timeout_time = 30  # 30 second timeout
        start_time = time.time()
        
        while self.robot_running:
            if self.target_pos_received is not None:
                diff = np.abs(np.array(self.target_pos_received[:self.dof]) - np.array(self.start_position))

                if np.all(diff < self.start_joint_tolerance):
                    self.logger_ri.info("Opening ceremony completed - target position within tolerance")
                    return True
            
            # Check timeout
            if time.time() - start_time > timeout_time:
                self.logger_ri.error("Opening ceremony timeout - target position not reached")
                return False
                
            time.sleep(self.control_dt)
        
        return False

    def _execute_play_recording(self):
        """
        Execute playback of recorded positions.
        """
        self.logger_ri.info("Starting play recording control loop.")
        while self.robot_running:
            start_time = time.time()
            
            if self.master_positions:
                action_deg = self.master_positions.popleft()
            else:
                self.logger_ri.info("control_loop, recording playbook completed")
                break

            # Safety check
            if not action_master_safety_check(action_deg, self.upper_limits, self.lower_limits):
                self.logger_ri.warning("Safety check failed for recorded position.")
                break

            # Send TMSCT command
            position = action_master_TM_translation(copy.deepcopy(action_deg))
            script_cmd = "Position({:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f})".format(*position)
            
            if not send_tmsct_cmd(self.sock, "2", script_cmd, self.logger_ri):
                self.logger_ri.error("Error sending command to Techman.")
                break

            self.logger_ri.info(f"Sent recorded position command: {script_cmd}")

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
                self.logger_ri.info(f"Sent stop command: {script_cmd}")
                # rate control
                start_time = time.time()
                elapsed = time.time() - start_time
                sleep_time = self.control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        self.receive_target_pos = False
        self.robot_running = False
        self.play_recording_active = False

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

        # Send movement command
        position = action_master_TM_translation(self.start_position)
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
            # Bind to all interfaces on the specified port to receive data from robot
            bind_address = ("", self.robot_recv_address[1])  # Empty string means all interfaces
            self.recv_sock.bind(bind_address)
            self.recv_sock.settimeout(0.5)
            self.logger_ri.info(f"Listener thread started. Listening on port {self.robot_recv_address[1]} for robot messages...")
            
            while self.listener_running:
                try:
                    data = self.recv_sock.recv(4096)
                    if data:
                        self.logger_ri.info(f"Received from Techman: {data.decode(errors='ignore')}")
                        # Process robot feedback here if needed
                except socket.timeout:
                    # Normal - no data arrived within 0.5s
                    pass
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

    def _store_joint_data(self, robot_position, master_position):
        """
        Store joint data in shared memory for recording.
        Gripper state should already be included as the last joint in position arrays.
        """
        robot_pos = robot_position.tolist() if isinstance(robot_position, np.ndarray) else robot_position
        master_pos = master_position.tolist() if isinstance(master_position, np.ndarray) else master_position
        
        joint_data = {
            "robot_position": robot_pos,
            "master_position": master_pos,
            "send_position_robot": master_pos,  # Use master as send position for TechMan
            "robot_position_timestamp": time.time(),
            "seq_id": getattr(self, 'seq_id', 0),
        }
        try:
            if self.shm_joint_data1.full():
                self.shm_joint_data1.get_nowait()
            self.shm_joint_data1.put(joint_data)
        except Exception as e:
            self.logger_ri.error(f"Could not store joint data: {e}")

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
            position = action_master_TM_translation(joint_position)
            script_cmd = "Position({:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f})".format(*position)
            return send_tmsct_cmd(self.sock, "1", script_cmd, self.logger_ri)
        except Exception as e:
            self.logger_ri.error(f"Error pushing joint motion: {e}")
            return False

    def get_joint_position(self):
        """
        Get current joint position (placeholder - implement if robot provides feedback).
        """
        # Return last known position or start position
        return self.start_position

    def get_joint_velocity(self):
        """
        Get current joint velocity (placeholder).
        """
        return [0.0] * self.dof

    def set_gripper_state(self, state):
        """
        Set gripper state (placeholder - implement if gripper control is needed).
        """
        self.gripper_state = state
        self.logger_ri.info(f"Gripper state set to: {state}")
        return True




##################################################################
# MAIN FUNCTION
##################################################################
def run_robot_interface(robot_interface_commup, robot_interface_commdown, shm_target_pos1, shm_target_pos2_info, shm_joint_data1, shm_joint_data2):
    """
    Main function to run the FanucRobot interface in a loop,
    checking for commands on 'robot_interface_commdown' queue.
    """
    component_tag = "ROBOT_INTERFACE"
    logger_ri = setup_logging(component_tag)

    logger_ri.info("Starting Robot Interface...")

    # Load the config
    config = load_config()
    logger_ri.info("Config loaded successfully.")

    # Robot brand name (e.g., 'Fanuc'), capitalized
    robot_brand = config["hardware"]["robot"]["brand"].lower().capitalize()
    check_queue_period = config["general"]["check_queue_period"]
    robot = None

    logger_ri.info(f"Starting {robot_brand} Robot Interface")

    # Dynamically instantiate the correct robot class
    try:
        robot_class_name = f"{robot_brand}Robot"
        robotClass = globals()[robot_class_name]
        robot = robotClass(robot_interface_commup, shm_target_pos1, shm_target_pos2_info, shm_joint_data1, shm_joint_data2, logger_ri, config)
        logger_ri.info("Initialized %s Robot Interface", robot_brand)
        # Send a success response for "initialization"
        send_response(
            logger_ri,
            robot_interface_commup,
            {"interface": component_tag, "message": "initialization"},
            error="None"
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
            error=f"{e}"
        )

    # Listen for commands in a loop
    while True:
        if not robot_interface_commdown.empty():
            full_message = robot_interface_commdown.get()
            logger_ri.info(f"Message received: {full_message}")

            msg_type = full_message.get("type", "")
            msg_interface = full_message.get("interface", "")
            message = full_message.get("message", "")

            if msg_type == "CMD" and msg_interface == "ROBOT_INTERFACE":
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
                else:
                    logger_ri.error("Unknown dict CMD message: %s", full_message)
                    send_response(logger_ri, robot_interface_commup, full_message,
                                  error="Unknown CMD message")
            else:
                logger_ri.error("Unknown dict message format: %s", full_message)
                send_response(logger_ri, robot_interface_commup, full_message,
                              error="Unknown dict message format")

        time.sleep(check_queue_period)
