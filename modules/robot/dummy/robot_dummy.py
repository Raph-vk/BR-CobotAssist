###################################################################
# Dummy Robot Implementation
###################################################################
import time
import numpy as np
import threading
from threading import Thread
import copy
from collections import deque
import json
import os
import sys
import multiprocessing.shared_memory as shared_memory
import struct

from ruckig import InputParameter, OutputParameter, Ruckig, Result, Synchronization

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from utils.utils import setup_logging, load_config


###################################################################
# Helper Functions (Global) - Same as FANUC robot
###################################################################
def send_tc_command(queue, command):
    """
    Helper function to put a command into a queue.
    """
    queue.put(command)


def J3_interaction(action):
    """
    J3 interaction function. Offsets the 3rd joint by subtracting
    the 2nd joint value from the 3rd.
    """
    action[2] = action[2] - action[1]
    return action


def J3_interaction_rev(action):
    """
    Reverse J3 interaction function. Offsets the 3rd joint by adding
    the 2nd joint value back to the 3rd.
    """
    action[2] = action[2] + action[1]
    return action


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
# DUMMY CONNECTION CLASSES
##################################################################
class DummyRMIConnection:
    """
    Dummy RMI connection that simulates the FANUC RMI interface
    without actual network communication.
    """
    def __init__(self, robot_address, rmi_port, logger, config):
        self.robot_address = robot_address
        self.rmi_port = rmi_port
        self.logger = logger
        self.config = config
        self.connected = False
        
    def connect(self):
        """Simulate connection - always succeeds."""
        self.logger.info("Dummy RMI: Simulating connection...")
        time.sleep(0.1)  # Simulate connection delay
        self.connected = True
        return True
        
    def disconnect(self):
        """Simulate disconnection."""
        self.logger.info("Dummy RMI: Simulating disconnection...")
        self.connected = False
        
    def reset(self):
        """Simulate reset - always succeeds."""
        self.logger.info("Dummy RMI: Simulating reset...")
        time.sleep(0.05)  # Simulate reset delay
        return True
        
    def initialize_rmi(self):
        """Simulate initialization - always succeeds."""
        self.logger.info("Dummy RMI: Simulating initialization...")
        time.sleep(0.05)  # Simulate init delay
        return True
        
    def push_joint_motion(self, position, speed=10, term_type="FINE", term_val=0):
        """Simulate joint motion command - always succeeds."""
        return True


class DummyUDPStreaming:
    """
    Dummy UDP streaming that simulates the FANUC UDP interface
    without actual network communication.
    """
    def __init__(self, logger, config, server_address, robot_address, control_dt, check_queue_period_divisor):
        self.logger = logger
        self.config = config
        self.server_address = server_address
        self.robot_address = robot_address
        self.control_dt = control_dt
        self.check_queue_period_divisor = check_queue_period_divisor
        
        # Simulate robot state
        self.current_position = [0.0] * 6  # Start at zero position
        self.current_velocity = [0.0] * 6
        self.target_position = [0.0] * 6
        self.gripper_state = 0
        self.seq_id_sent = 0
        self.seq_id_recv = 0
        
        # UDP socket simulation
        self.socket_setup = False
        
        # State flags
        self.running = False
        self.streaming_thread = None
        
    def setup_udp_socket(self):
        """Simulate UDP socket setup - always succeeds."""
        self.logger.info("Dummy UDP: Simulating socket setup...")
        time.sleep(0.01)
        self.socket_setup = True
        return True
        
    def close_socket(self):
        """Simulate socket close."""
        self.logger.info("Dummy UDP: Simulating socket close...")
        self.socket_setup = False
        self.running = False
        
    def send_stop_packet(self):
        """Simulate stop packet."""
        self.logger.info("Dummy UDP: Simulating stop packet...")
        self.running = False
        
    def create_limit_table(self, payload_weight, robot_max_payload):
        """Simulate limit table creation - always succeeds."""
        self.logger.info(f"Dummy UDP: Simulating limit table creation (payload: {payload_weight}, max: {robot_max_payload})")
        time.sleep(0.02)
        return True
        
    def start_status_streaming(self):
        """Start the dummy status streaming thread."""
        if self.streaming_thread and self.streaming_thread.is_alive():
            return True
            
        self.running = True
        self.streaming_thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self.streaming_thread.start()
        self.logger.info("Dummy UDP: Started status streaming thread")
        return True
        
    def stop_status_streaming(self):
        """Stop the dummy status streaming thread."""
        self.running = False
        if self.streaming_thread:
            self.streaming_thread.join(timeout=1.0)
            self.streaming_thread = None
        self.logger.info("Dummy UDP: Stopped status streaming thread")
        
    def _streaming_loop(self):
        """Simulate the status streaming loop."""
        while self.running:
            # Simulate gradual movement towards target position
            for i in range(len(self.current_position)):
                if i < len(self.target_position):  # Safety check to prevent index out of range
                    diff = self.target_position[i] - self.current_position[i]
                    if abs(diff) > 0.001:  # Small threshold to avoid infinite movement
                        # Move 10% towards target each step
                        self.current_position[i] += diff * 0.1
                        self.current_velocity[i] = diff * 0.1 / self.control_dt
                    else:
                        self.current_velocity[i] = 0.0
                        
            time.sleep(self.control_dt)
            
    def send_position_udp(self, position, gripper_state=False):
        """Simulate sending position via UDP."""
        self.target_position = position[:6]  # Only take first 6 elements for joint positions
        self.gripper_state = gripper_state
        self.seq_id_sent += 1
        return True
        
    def send_joint_pos(self, position, gripper_on, gripper_off):
        """Simulate sending joint position via UDP - matches FANUC interface."""
        self.target_position = position[:6]  # Only take first 6 elements for joint positions
        
        # Update gripper state based on gripper_on/gripper_off flags
        if gripper_on:
            self.gripper_state = True
        elif gripper_off:
            self.gripper_state = False
            
        self.seq_id_sent += 1
        return True
        
    def get_current_position(self):
        """Get the current simulated position."""
        return self.current_position.copy()
        
    def get_current_velocity(self):
        """Get the current simulated velocity."""
        return self.current_velocity.copy()


##################################################################
# CLASS DEFINITION: DummyRobot
##################################################################
class DummyRobot:
    """
    Dummy robot implementation that mimics the FanucRobot class interface
    but without any actual robot communication. This allows testing of the
    full system without a physical robot.
    """

    def __init__(self, robot_interface_commup, shm_target_pos1, shm_target_pos2_info, shm_joint_data1, shm_joint_data2, logger_ri, config):
        # Logging and config - set these first
        self.logger_ri = logger_ri
        self.config = config
        self.record_duration = config["general"]["record_duration"]

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
                self.logger_ri.info(f"Dummy: Attached to shm_target_pos2: {shm_target_pos2_info['name']}")
            except Exception as e:
                self.logger_ri.error(f"Dummy: Failed to attach to shm_target_pos2: {e}")
                self.shm_target_pos2 = None

        # Config elements
        self.utool_no = config["hardware"]["robot"]["utool_no"]
        self.uframe_no = config["hardware"]["robot"]["uframe_no"]
        self.status_refresh_period = config["general"]["status_refresh_period"]
        self.check_queue_period = config["general"]["check_queue_period"]
        self.control_dt = config["hardware"]["robot"]["control_dt"]
        self.check_queue_period_divisor = config["general"]["check_queue_period_divisor"]
        self.payload_weight = config["hardware"]["robot"]["payload_weight"]
        self.robot_max_payload = config["hardware"]["robot"]["max_payload"]
        self.action_buffer_length = config["general"]["action_buffer_length"]
        self.start_position = config["general"]["start_position"]
        self.joint3_limit = config["hardware"]["robot"]["j3_limit"]
        self.joint3_interaction = config["hardware"]["robot"]["j3_interaction"]
        self.start_joint_tolerance = config["general"]["start_joint_tolerance"]
        self.joint_synchronization = config["general"]["joint_synchronization"]
        self.dof = config["hardware"]["robot"]["dof"]
        self.default_recording_speed = config["general"]["default_recording_speed"]
        self.upper_limits = config["hardware"]["robot"]["upper_limits"]
        self.lower_limits = config["hardware"]["robot"]["lower_limits"]
        self.gripper_treshold = config["general"]["gripper_treshold"]
        self.gripper_delay = config["general"]["gripper_delay"]
        self.control_loop_language = config["general"]["control_loop_language"]
        self.robot_speed = self.default_recording_speed

        # State flags
        self.messages_sent = 0
        self.vacuum_on = True
        self.last_vacuum_toggle_step = 0
        self.total_timesteps = 0
        self.counter = 0
        self.recording = False
        self.gripper_state = 0
        self.gripper_state_change_time = 0
        self.gripper_on = False
        self.gripper_off = False
        self.master_positions = deque()

        # Dummy connection settings
        self.rmi_port = config["hardware"]["robot"]["rmi_port"]
        self.rmi_seq_id = 0
        self.server_address = config["hardware"]["robot"]["server_address"]
        self.robot_address = config["hardware"]["robot"]["robot_address"]

        # Create dummy connection objects
        self.rmi = DummyRMIConnection(
            robot_address=self.robot_address,
            rmi_port=self.rmi_port,
            logger=self.logger_ri,
            config=self.config
        )

        self.udp = DummyUDPStreaming(
            logger=self.logger_ri,
            config=self.config,
            server_address=self.server_address,
            robot_address=self.robot_address,
            control_dt=self.control_dt,
            check_queue_period_divisor=self.check_queue_period_divisor
        )

        # Connection flags
        self.connected = False
        self.rmi_connected = False
        self.play_recording_active = False
        self.run_policy_active = False

        # Threads & flags
        self.status_streaming = False
        self.receive_target_pos = False
        self.robot_running = False
        
        # Thread references
        self.update_target_info_thread = None
        self.robot_control_thread = None
        
        # Additional state variables needed for dummy implementation
        self.target_pos_received = self.start_position.copy()
        self.robot_speed = self.default_recording_speed
        self.first_joint_position = self.start_position.copy()  # For play_recording start position check
        self.joint_limits = None
        
        self.logger_ri.info("Dummy robot initialized successfully")

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
                self.logger_ri.error("Dummy: start_teleoperation, could not connect to robot")
                return False

        # Create thread that generates target positions
        self.update_target_info_thread = threading.Thread(
            target=self._update_target_info, args=(full_message, ), daemon=True
        )
        self.update_target_info_thread.start()
        self.logger_ri.info("Dummy: start_teleoperation, target position updating started")

        # create thread that controls the robot
        self.robot_control_thread = threading.Thread(
            target=self._control_robot, args=(full_message, ), daemon=True
        )
        self.robot_control_thread.start()
        self.logger_ri.info("Dummy: start_teleoperation, robot control thread created")
        return True

    def start_teleoperation_record(self, full_message):
        """
        Same as start_teleoperation, but also sets self.recording=True
        so we store positions in shm_joint_data1.
        """
        self.recording = True
        return self.start_teleoperation(full_message)

    def record_episodes(self, full_message):
        self.recording = True
        return self.start_teleoperation(full_message)

    def run_policy(self, full_message):
        """
        Run the policy execution.
        """
        self.robot_running = True
        self.run_policy_active = True

        # Ensure robot is connected
        if not self.connected:
            if not self.connect():
                self.logger_ri.error("Dummy: run_policy, could not connect to robot")
                return False

        # Set robot speed for policy execution
        self.robot_speed = self.default_recording_speed

        # create thread that controls the robot
        self.robot_control_thread = threading.Thread(
            target=self._control_robot, args=(full_message, ), daemon=True
        )
        self.robot_control_thread.start()
        self.logger_ri.info("Dummy: run_policy, robot control thread created for policy execution")
        return True

    def play_recording(self, full_message):
        """
        Play the recorded positions from a JSON file.
        """
        self.robot_running = True
        self.play_recording_active = True

        # Ensure robot is connected
        if not self.connected:
            if not self.connect():
                self.logger_ri.error("Dummy: play_recording, could not connect to robot")
                return False

        # Create thread that generates target positions
        self.update_target_info_thread = threading.Thread(
            target=self._update_target_info, args=(full_message, ), daemon=True
        )
        self.update_target_info_thread.start()
        self.logger_ri.info("Dummy: play_recording, target position updating started")

        # create thread that controls the robot
        self.robot_control_thread = threading.Thread(
            target=self._control_robot, args=(full_message, ), daemon=True
        )
        self.robot_control_thread.start()
        self.logger_ri.info("Dummy: play_recording, robot control thread created")
        return True

    def stop(self):
        """
        Stop teleoperation or recording if running,
        stop UDP streaming, then call RMIConnection.disconnect().
        """
        if not self.connected:
            self.logger_ri.info("Dummy: stop, already disconnected")
            return True

        # Stop robot if running
        self.logger_ri.info("Dummy: stop, stopping robot running")
        if self.robot_running:
            self.stop_robot_running()
        self.logger_ri.info("Dummy: stop, robot running stopped")

        # Stop UDP streaming
        self.logger_ri.info("Dummy: stop, stopping UDP streaming")
        self.udp.send_stop_packet()
        self.udp.close_socket()
        self.logger_ri.info("Dummy: stop, UDP streaming stopped")

        # Disconnect RMI
        self.logger_ri.info("Dummy: stop, stopping RMI")
        self.rmi.disconnect()
        self.logger_ri.info("Dummy: stop, RMI stopped")

        self.connected = False
        self.rmi_connected = False
        self.recording = False
        return True

    def connect(self):
        """
        1) Uses DummyRMIConnection to simulate the TCP handshake.
        2) Sets up dummy UDP socket.
        3) Resets, initializes RMI.
        4) Creates limit table.
        """
        if self.connected:
            self.logger_ri.info("Dummy: connect, Already connected")
            return True

        # STEP 1: Connect RMI
        if not self.rmi.connect():
            self.logger_ri.error("Dummy: connect, RMI connect failed")
            self._cleanup_connection()
            return False
        self.rmi_connected = True

        # STEP 2: Set up the UDP socket
        if not self.udp.setup_udp_socket():
            self.logger_ri.error("Dummy: connect, Could not set up UDP socket")
            self._cleanup_connection()
            return False
        self.logger_ri.info("Dummy: connect, UDP socket set up")

        # STEP 3: RMI reset
        if not self.rmi.reset():
            self.logger_ri.error("Dummy: connect, RMI reset failed")
            self._cleanup_connection()
            return False
        self.logger_ri.info("Dummy: connect, RMI reset")

        # STEP 4: RMI initialize
        if not self.rmi.initialize_rmi():
            self.rmi.reset()
            self.logger_ri.error("Dummy: connect, RMI initialize failed")
            self._cleanup_connection()
            return False
        self.logger_ri.info("Dummy: connect, RMI initialized")

        # STEP 5: Fill limit tables
        if not self.udp.create_limit_table(self.payload_weight, self.robot_max_payload):
            self.logger_ri.error("Dummy: connect, create_limit_table failed")
            self._cleanup_connection()
            return False
        self.logger_ri.info("Dummy: connect, limit tables created")

        self.logger_ri.info("Dummy: connect, all steps successful. Robot is connected")
        self.connected = True
        return True

    def _cleanup_connection(self):
        """
        Force-close all partial connections/sockets/threads
        so the object is ready for a fresh reconnect attempt.
        """
        # Stop any streaming that might be running
        self.udp.send_stop_packet()
        self.udp.close_socket()

        # Disconnect RMI forcibly
        self.rmi.disconnect()

        # Reset flags
        self.connected = False
        self.rmi_connected = False
        self.recording = False
        self.play_recording_active = False
        self.run_policy_active = False
        
        self.robot_running = False
        self.receive_target_pos = False

        # Join threads if alive
        if self.update_target_info_thread:
            try:
                self.update_target_info_thread.join(timeout=1.0)
            except:
                pass
            self.update_target_info_thread = None

        if self.robot_control_thread:
            try:
                self.robot_control_thread.join(timeout=1.0)
            except:
                pass
            self.robot_control_thread = None

        self.logger_ri.info("Dummy: _cleanup_connection: All connections closed; ready to reconnect")

    ################################################################
    # MAIN LOGIC (Threads, Teleoperation Control, Recording Playback)
    ################################################################

    def _get_next_policy_action(self):
        """
        Read the next target action from shm_target_pos2 based on the last seq_id sent to robot.
        Uses optimized search: first tries modulo position, then searches if needed.
        Returns the joint positions for the next sequence ID, or None if not available.
        """
        if not self.shm_target_pos2:
            return None
        
        try:
            # Calculate the next sequence ID we need
            next_seq_id = self.udp.seq_id_sent + 1
            
            def read_buffer_entry(buffer_index):
                """Helper to read and unpack data at a specific buffer index."""
                offset = buffer_index * self.shm_target_pos2_entry_size
                data = self.shm_target_pos2.buf[offset:offset+self.shm_target_pos2_entry_size]
                return struct.unpack(self.shm_target_pos2_entry_format, data)
            
            # First, try the expected modulo position (fast path)
            expected_index = next_seq_id % self.shm_target_pos2_capacity
            unpacked_data = read_buffer_entry(expected_index)
            seq_id = unpacked_data[0]
            
            if seq_id == next_seq_id:
                joint_positions = list(unpacked_data[1:])
                return joint_positions
            
            for buffer_index in range(self.shm_target_pos2_capacity):
                if buffer_index == expected_index:
                    continue  # Already checked this one
                    
                unpacked_data = read_buffer_entry(buffer_index)
                seq_id = unpacked_data[0]
                
                if seq_id == next_seq_id:
                    joint_positions = list(unpacked_data[1:])
                    return joint_positions
            
            return None
            
        except Exception as e:
            self.logger_ri.error(f"Dummy: Error reading policy action from shared memory: {e}")
            return None

    def _update_target_info(self, full_message):
        """
        Thread that continuously reads from shm_target_pos1
        to update self.target_pos_received.
        """
        message = full_message.get("message", "")

        if message == "start_teleoperation" or message == "start_teleoperation_record" or message == "record_episodes":
            # Check if recording_speed is valid, otherwise use default
            recording_speed = full_message.get("recording_speed", self.default_recording_speed)
            if recording_speed == "" or recording_speed is None:
                self.robot_speed = self.default_recording_speed
                self.logger_ri.warning(
                    "Dummy: _update_target_info: Empty or null recording_speed, using default: %s",
                    self.default_recording_speed
                )
            else:
                self.robot_speed = recording_speed
                self.logger_ri.info("Dummy: _update_target_info, recording_speed: %s", self.robot_speed)

            # Empty the queue of the shm_target_pos1
            while not self.shm_target_pos1.empty():
                try:
                    self.shm_target_pos1.get(timeout=0.01)
                except Exception:
                    pass

            # Start the loop to receive target positions
            while self.receive_target_pos:
                try:
                    target_pos = self.shm_target_pos1.get(timeout=1)
                    self.target_pos_received = target_pos

                    if self.control_loop_language == "cpp":
                        self.target_pos_received = target_pos.tolist()
                except Exception as e:
                    self.logger_ri.warning("Dummy: _update_target_info, no position received: %s", e)
                    pass
                time.sleep(self.control_dt / self.check_queue_period_divisor)

        if message == "play_recording":
            # Load recording data
            try:
                filename = full_message["recording_name"]
                filename = os.path.join("data", filename)
                if not os.path.exists(filename):
                    self.logger_ri.error("Dummy: play_recording, file %s not found.", filename)
                    return False
            except KeyError:
                self.logger_ri.error("Dummy: _update_target_info, missing recording_name in message.")
                return False

            # Load JSON
            with open(filename, 'r') as f:
                data = json.load(f)
                self.master_positions = deque([s["master_position"] for s in data["samples"]])
                try:
                    self.first_joint_position = data["samples"][0]["robot_position"]
                except IndexError:
                    self.logger_ri.error("Dummy: _update_target_info, no robot positions found in %s.", filename)
                    return False

            if not data:
                self.logger_ri.error("Dummy: _update_target_info, %s is empty.", filename)
                return False

            # Check playback_speed
            self.default_recording_speed = data["metadata"]["recording_speed"]
            playback_speed = full_message.get("playback_speed", "")
            if playback_speed == "" or playback_speed is None:
                self.robot_speed = self.default_recording_speed
                self.logger_ri.info(
                    "Dummy: _update_target_info: Empty or null playback_speed, using default: %s",
                    self.default_recording_speed
                )
            else:
                self.robot_speed = playback_speed * self.default_recording_speed
                self.logger_ri.info("Dummy: _update_target_info, playback_speed: %s", self.robot_speed)

    def _control_robot(self, full_message):
        """
        Main control thread for teleoperation or playback.

        1) Move to start position.
        2) Check start position tolerance or relevant conditions.
        3) Prepare robot (call TPP program, start UDP streaming).
        4) Control loop (Ruckig trajectory calculation, sending joint pos).
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
        position = copy.deepcopy(self.start_position)
        if not self.push_joint_motion(position, speed=10, term_type="FINE", term_val=0):
            self.logger_ri.error("Dummy: _move_to_start_position, could not move to start position.")
            return False
        self.udp.current_position = self.start_position[:-1]  # Update dummy position (only joint positions)
        # Extract gripper state from start_position (last element)
        gripper_state = 1 if (len(self.start_position) > self.dof and self.start_position[-1] > 0.5) else 0
        self.udp.gripper_state = gripper_state
        self.logger_ri.info("Dummy: _move_to_start_position, moved to start position.")
        return True

    def _check_start_position(self, full_message):
        """
        Verify that the current position is near the start position,
        or handle 'opening_ceremony' if teleoperation is started.
        """
        message = full_message.get("message", "")

        if message == "start_teleoperation" or message == "start_teleoperation_record" or message == "record_episodes":
            self.logger_ri.info("Dummy: checking start position")
            if not self.opening_ceremony():
                self.logger_ri.error("Dummy: _check_start_position, opening ceremony failed")
                return False

        if message == "play_recording":
            # Check if start position is close to the first recording position
            diffs = [abs(c - s) for c, s in zip(self.start_position, self.first_joint_position)]
            if any(d > self.start_joint_tolerance for d in diffs):
                self.logger_ri.error(
                    "Dummy: _check_start_position, start position is not close to the first position in the recording"
                )
                return False
            self.logger_ri.info("Dummy: _check_start_position, checking start position")
            return True

    def _prepare_robot(self, full_message):
        """
        Make the robot ready for streaming:
        - Call TPP program via RMI (dummy)
        - Start UDP streaming (dummy)
        """
        # Simulate determining robot limits
        current_speed = 200
        # For dummy, just create some reasonable joint limits
        self.joint_limits = [[self.lower_limits[i], self.upper_limits[i]] for i in range(self.dof)]
        
        started_streaming = False
        if self.robot_running:
            # Simulate calling the TPP program
            self.logger_ri.info("Dummy: _prepare_robot, simulating motion stream call")
            time.sleep(0.01)  # Simulate processing time

            self.logger_ri.info("Dummy: _prepare_robot, starting streaming")
            self.udp.start_status_streaming()  # Start our dummy streaming
            started_streaming = True

        return started_streaming

    def _control_loop(self, started_streaming):
        """
        Core loop for either teleoperation or playing back a recorded trajectory:
        - Setup Ruckig
        - Fill action buffer
        - While running, compute next motion using Ruckig and send to robot
        - Optionally store motion data if recording
        """
        if self.control_loop_language == "python":
            self.logger_ri.info("Dummy: control_loop, using Python control loop")
            
            # Setup Ruckig for trajectory planning
            otg, inp, out = self.setup_ruckig()
            
            # Initialize positions - get current position from UDP (6 elements)
            current_position = self.udp.get_current_position()
            previous_action_master = self.start_position
            
            # Adjust process priority (dummy)
            self.adjust_process_priority(-10)
            
            # Fill action buffer with start position (like FANUC robot)
            self.logger_ri.info("Dummy: control_loop, streaming started, filling buffer")
            for _ in range(self.action_buffer_length):
                # Extract gripper state from start_position (last element)
                gripper_state = 1 if (len(self.start_position) > self.dof and self.start_position[-1] > 0.5) else 0
                self.udp.send_joint_pos(self.start_position[:-1], gripper_state, False)

            self.logger_ri.info("Dummy: control_loop, starting control loop")
            previous_action_master = self.start_position
            
            step = 0
            self.logger_ri.info(f"Dummy: Starting control loop. Recording={self.recording}, Policy={self.run_policy_active}, Playback={self.play_recording_active}")
            while self.robot_running:
                try:
                    # Get next target action based on the current mode
                    if self.run_policy_active:
                        action_master = self._get_next_policy_action()
                        if action_master is None:
                            time.sleep(self.control_dt)
                            continue
                    elif self.play_recording_active and self.master_positions:
                        # Get next position from recording
                        if self.master_positions:
                            action_master = self.master_positions.popleft()
                        else:
                            # Recording finished
                            self.logger_ri.info("Dummy: control_loop, recording playback finished")
                            break
                    else:
                        # For teleoperation, get from target_pos_received
                        action_master = getattr(self, 'target_pos_received', current_position.copy())
                        
                    # Apply J3 interaction if enabled
                    if self.joint3_interaction:
                        action_master = J3_interaction(action_master)
                    
                    # Update Ruckig input
                    self.update_ruckig_input(action_master, inp, previous_action_master)
                    
                    # Calculate trajectory
                    new_position = self.trajectory_calculation(otg, inp, out, current_position)
                    
                    # Determine gripper state based on action_master
                    self.determine_gripper_state(action_master[-1])
                    
                    # Apply J3 interaction reverse if enabled
                    if self.joint3_interaction:
                        new_position = J3_interaction_rev(new_position)
                    
                    # Store flags before sending
                    gripper_on_to_send = self.gripper_on
                    gripper_off_to_send = self.gripper_off
                    
                    # Send position to dummy robot using joint_pos method
                    success_send = self.udp.send_joint_pos(new_position, gripper_on_to_send, gripper_off_to_send)
                    
                    # Reset gripper flags after sending
                    self.gripper_on = False
                    self.gripper_off = False
                    
                    if not success_send:
                        self.logger_ri.error("Dummy: control_loop, could not send joint position to robot")
                        break
                    
                    # Record if recording is active (same as FANUC robot)
                    if self.recording:
                        try:
                            if self.shm_joint_data1.full():
                                self.shm_joint_data1.get_nowait()
                            
                            # Convert gripper state to float (0.0 or 1.0) and append to position arrays
                            gripper_state_float = 1.0 if self.gripper_state else 0.0
                            master_pos_with_gripper = action_master + [gripper_state_float]
                            send_pos_with_gripper = new_position + [gripper_state_float]
                            last_received_js = new_position + [gripper_state_float]  # Simulate received joint state
                            
                            joint_data = {
                                "master_position": master_pos_with_gripper,
                                "send_position_robot": send_pos_with_gripper,
                                "robot_position": last_received_js,  # Already includes gripper state
                                "robot_position_timestamp": time.time(),
                                "seq_id": self.udp.seq_id_sent,
                            }
                            self.shm_joint_data1.put(joint_data)
                        except Exception as e:
                            self.logger_ri.error("Dummy: error uploading joint data: %s", e)
                    
                    # Record for policy execution if active (same as FANUC robot)
                    if self.run_policy_active:
                        try:
                            if self.shm_joint_data2.full():
                                self.shm_joint_data2.get_nowait()
                            
                            # Convert gripper state to float (0.0 or 1.0) and append to position arrays
                            gripper_state_float = 1.0 if self.gripper_state else 0.0
                            master_pos_with_gripper = action_master + [gripper_state_float]
                            send_pos_with_gripper = new_position + [gripper_state_float]
                            last_received_js = new_position + [gripper_state_float]  # Simulate received joint state
                            
                            def _to_list_if_array(x):
                                import numpy as np
                                return x.tolist() if isinstance(x, np.ndarray) else x

                            joint_data = {
                                "master_position": _to_list_if_array(master_pos_with_gripper),
                                "send_position_robot": _to_list_if_array(send_pos_with_gripper),
                                "robot_position": _to_list_if_array(last_received_js),
                                "robot_position_timestamp": time.time(),  # or time.time() in dummy
                                "seq_id": self.udp.seq_id_sent,
                            }
                            self.shm_joint_data2.put(joint_data)
                        except Exception as e:
                            self.logger_ri.error("Dummy: error uploading policy joint data: %s", e)
                    
                    # Update state
                    current_position = new_position.copy()
                    previous_action_master = action_master.copy()
                    
                    step += 1
                    time.sleep(self.control_dt)
                    
                except Exception as e:
                    self.logger_ri.error(f"Dummy: Error in control loop step {step}: {e}")
                    time.sleep(self.control_dt)
                    continue
                    
        elif self.control_loop_language == "cpp":
            self.logger_ri.info("Dummy: control_loop, would use C++ control loop (not implemented in dummy)")
            # For dummy, just simulate the control loop
            step = 0
            while self.robot_running:
                time.sleep(self.control_dt)
                step += 1
        else:
            self.logger_ri.error(
                "Dummy: control_loop, unsupported control loop language: %s", self.control_loop_language
            )

    def _close_robot(self, started_streaming):
        """
        Close out any running streaming after control loop ends.
        """
        if started_streaming:
            self.logger_ri.info("Dummy: _close_robot, stopping UDP streaming")
            self.udp.send_stop_packet()
            self.udp.stop_status_streaming()
            self.logger_ri.info("Dummy: _close_robot, UDP streaming stopped")

    def _send_stop_response(self, full_message):
        """
        If the robot was playing a recording, send a 'stop' command
        back to the queue to indicate the operation completed.
        """
        message = full_message.get("message", "")
        if message == "play_recording":
            send_tc_command(self.robot_interface_commup, {"type": "CMD", "message": "stop", "interface": "ROBOT_INTERFACE"})
            self.logger_ri.info("Dummy: _send_stop_response, sent stop command to queue.")

    ################################################################
    # CRITICAL METHODS MATCHING FANUC ROBOT IMPLEMENTATION
    ################################################################

    def push_joint_motion(self, position, speed=10, term_type="FINE", term_val=0):
        """
        Wrapper around DummyRMIConnection.push_joint_motion.
        Applies J3_interaction if configured, then calls the RMI method.
        """
        if self.joint3_interaction:
            position = J3_interaction(position)
        return self.rmi.push_joint_motion(position, speed, term_type, term_val)

    def stop_robot_running(self):
        """
        Stop the teleoperation or play_recording threads and reset relevant flags.
        """
        self.receive_target_pos = False
        self.robot_running = False
        self.play_recording_active = False
        self.run_policy_active = False
        
        # Join threads
        if self.update_target_info_thread and self.update_target_info_thread.is_alive():
            self.update_target_info_thread.join(timeout=2.0)
        if self.robot_control_thread and self.robot_control_thread.is_alive():
            self.robot_control_thread.join(timeout=2.0)

    def opening_ceremony(self):
        """
        Dummy opening ceremony - just log that we're ready.
        """
        self.logger_ri.info("Dummy: Opening ceremony - robot is ready!")
        return True

    def setup_ruckig(self):
        """
        Set up Ruckig trajectory generator with robot parameters.
        """
        self.logger_ri.info("Dummy: Setting up Ruckig trajectory generator")
        
        # Create Ruckig instance
        otg = Ruckig(self.dof, self.control_dt)
        inp = InputParameter(self.dof)
        out = OutputParameter(self.dof)
        
        # Set limits from config
        inp.max_velocity = [50.0] * self.dof  # degrees/second
        inp.max_acceleration = [100.0] * self.dof  # degrees/second^2
        inp.max_jerk = [200.0] * self.dof  # degrees/second^3
        
        # Initialize with start position (only joint positions, not gripper)
        inp.current_position = self.start_position[:self.dof]
        inp.current_velocity = [0.0] * self.dof
        inp.current_acceleration = [0.0] * self.dof
        
        inp.target_position = self.start_position[:self.dof]
        inp.target_velocity = [0.0] * self.dof
        inp.target_acceleration = [0.0] * self.dof
        
        return otg, inp, out

    def stop_robot(self, otg, inp, out):
        """
        Generate a stopping trajectory using Ruckig.
        """
        # Set target to current position to stop smoothly
        inp.target_position = inp.current_position.copy()
        inp.target_velocity = [0.0] * self.dof
        inp.target_acceleration = [0.0] * self.dof
        
        return inp.current_position.copy()

    def update_ruckig_input(self, action_master, inp, previous_action_master):
        """
        Update Ruckig input parameters with new target action.
        """
        # Set new target - only use first 6 elements for joint positions
        inp.target_position = action_master[:self.dof]
        inp.target_velocity = [0.0] * self.dof
        inp.target_acceleration = [0.0] * self.dof
        
        # Apply joint synchronization if enabled
        if self.joint_synchronization:
            inp.synchronization = Synchronization.TimeSync

    def trajectory_calculation(self, otg, inp, out, current_position):
        """
        Calculate the next trajectory point using Ruckig.
        """
        try:
            # Update current state
            inp.current_position = current_position.copy()
            
            # Calculate next step
            result = otg.update(inp, out)
            
            if result == Result.Working or result == Result.Finished:
                # Update input for next iteration
                inp.current_position = out.new_position
                inp.current_velocity = out.new_velocity
                inp.current_acceleration = out.new_acceleration
                
                return list(out.new_position)
            else:
                # If trajectory calculation fails, return current position
                self.logger_ri.warning(f"Dummy: Ruckig calculation failed with result: {result}")
                return current_position.copy()
                
        except Exception as e:
            self.logger_ri.error(f"Dummy: Error in trajectory calculation: {e}")
            return current_position.copy()

    def handle_trajectory_error(self, e, inp, out, otg, dof):
        """
        Handle trajectory calculation errors.
        """
        self.logger_ri.error(f"Dummy: Trajectory error: {e}")
        # Return current position as fallback
        return inp.current_position.copy()

    def determine_gripper_state(self, gripper_state):
        """
        Decide when to activate / deactivate the gripper based on threshold and
        internal 'gripper_delay' logic. Sets gripper_on/gripper_off flags for UDP sending.
        """
        now = time.time()
        self.gripper_on = False
        self.gripper_off = False
        
        # Turn ON
        if gripper_state >= self.gripper_treshold and not self.gripper_state:
            if self.gripper_delay > 0.0:
                self.gripper_state_change_time_threshold = self.gripper_delay / self.robot_speed
                if (self.gripper_state_change_time + self.gripper_state_change_time_threshold) > now:
                    self.gripper_on = True
                    self.gripper_state = 1
            else:
                self.gripper_on = True
                self.gripper_state = 1

        # Turn OFF
        elif gripper_state < self.gripper_treshold and self.gripper_state:
            self.gripper_off = True
            self.gripper_state = 0
            self.gripper_state_change_time = now

    def adjust_process_priority(self, priority):
        """
        Adjust process priority (dummy implementation).
        """
        self.logger_ri.info(f"Dummy: Simulating process priority adjustment to {priority}")


def run_robot_interface(robot_interface_commup, robot_interface_commdown, shm_target_pos1, shm_target_pos2_info, shm_joint_data1, shm_joint_data2):
    """
    Main function to run the DummyRobot interface in a loop,
    checking for commands on 'robot_interface_commdown' queue.
    """
    component_tag = "ROBOT_INTERFACE"
    logger_ri = setup_logging(component_tag)

    logger_ri.info("Starting Robot Interface...")

    # Load the config
    config = load_config()
    logger_ri.info("Config loaded successfully.")

    # Robot brand name (e.g., 'Dummy'), capitalized
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
