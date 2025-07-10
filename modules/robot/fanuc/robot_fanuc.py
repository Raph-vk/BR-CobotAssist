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
import gc

from ruckig import InputParameter, OutputParameter, Ruckig, Result, Synchronization

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from utils.utils import setup_logging, load_config, get_data_path
from .robot_fanuc_rmi import RMIConnection
from .robot_fanuc_udp import UDPStreaming

BUILD_DIR = os.path.join(os.path.dirname(__file__), "cpp", "build")
sys.path.append(BUILD_DIR)
import control_loop_module


###################################################################
# Helper Functions (Global)
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
# CLASS DEFINITION: FanucRobot
##################################################################
class FanucRobot():
    """
    Handles:
      - RMI connection (TCP) to a FANUC robot (delegated to RMIConnection),
      - A separate UDPStreaming object for status streaming and limit-table logic,
      - Methods for motion commands (push_joint_motion, call_program),
      - Teleoperation, recording, playback,
      - Reading/writing I/O,
      - Additional logic previously from "RobotServer" code.
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
                self.logger_ri.info(f"Attached to shm_target_pos2: {shm_target_pos2_info['name']}")
            except Exception as e:
                self.logger_ri.error(f"Failed to attach to shm_target_pos2: {e}")
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

        # RMI Connection object
        self.rmi_port = config["hardware"]["robot"]["rmi_port"]
        self.rmi_seq_id = 0
        self.server_address = config["hardware"]["robot"]["server_address"]
        self.robot_address = config["hardware"]["robot"]["robot_address"]

        # Create an RMIConnection instance
        self.rmi = RMIConnection(
            robot_address=self.robot_address,
            rmi_port=self.rmi_port,
            logger=self.logger_ri,
            config=self.config
        )

        # Create our UDP streaming object
        self.udp = UDPStreaming(
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
        self.status_thread = None
        self.update_target_info_thread = None
        self.robot_control_thread = None
        self.cpp_obj = None

        # Kinematic states
        self.target_pos_received = None

        # Connect automatically if desired:
        if not self.connect():
            raise Exception(
                "FanucRobot: Could not connect to robot. Check IP address, connection, and teachpendant state."
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
                    "play_recording, could not connect to robot. "
                    "Check IP address, connection, or teachpendant state."
                )
            self.logger_ri.error("play_recording, robot not connected.")
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
    
    def record_episode(self, full_message):
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
                self.logger_ri.error(
                    "play_recording, could not connect to robot. "
                    "Check IP address, connection, or teachpendant state."
                )
            self.logger_ri.error("play_recording, robot not connected.")
            return False

        # Set robot speed for policy execution from model config
        model_name = full_message.get('model_name')
        dataset_name = full_message.get('dataset_name')
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


        # create thread that controls the robot
        self.robot_control_thread = threading.Thread(
            target=self._control_robot, args=(full_message, ), daemon=True
        )
        self.robot_control_thread.start()
        self.logger_ri.info("run_policy, robot control thread created for policy execution.")
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
                self.logger_ri.error(
                    "play_recording, could not connect to robot. "
                    "Check IP address, connection, or teachpendant state."
                )
            self.logger_ri.error("play_recording, robot not connected.")
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
        stop UDP streaming, then call RMIConnection.disconnect().
        """
        if not self.connected:
            self.logger_ri.info("stop, already disconnected.")
            return True

        # Stop robot if running
        self.logger_ri.info("stop, stopping robot running.")
        if self.robot_running:
            self.stop_robot_running()
        self.logger_ri.info("stop, robot running stopped.")

        # Stop UDP streaming
        self.logger_ri.info("stop, stopping UDP streaming.")
        self.udp.send_stop_packet()
        self.udp.close_socket()
        self.logger_ri.info("stop, UDP streaming stopped.")

        # Disconnect RMI
        self.logger_ri.info("stop, stopping RMI.")
        self.rmi.disconnect()
        self.logger_ri.info("stop, RMI stopped.")

        self.connected = False
        self.rmi_connected = False
        self.recording = False
        return True

    ################################################################
    # 2) MAIN LOGIC (Threads, Teleoperation Control, Recording Playback)
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
            
            # If we get here, the seq_id we need wasn't found in the buffer
            return None
                
        except Exception as e:
            self.logger_ri.error(f"Failed to read from shm_target_pos2: {e}")
            return None

    def _update_target_info(self, full_message):
        """
        Thread that continuously reads from shm_target_pos1
        to update self.target_pos_received.
        """
        message = full_message.get("message", "")

        if message == "start_teleoperation" or message == "start_teleoperation_record" or message == "record_episodes" or message == "record_episode":
            # Check if recording_speed is valid, otherwise use default
            recording_speed = full_message.get("recording_speed", self.default_recording_speed)
            if recording_speed == "" or recording_speed is None:
                self.robot_speed = self.default_recording_speed
                self.logger_ri.warning(
                    "_update_target_info: Empty or null recording_speed, using default: %s",
                    self.default_recording_speed
                )
            else:
                self.robot_speed = recording_speed
                self.logger_ri.info("_update_target_info, recording_speed: %s", self.robot_speed)

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
                        # call cpp function to 
                        if self.cpp_obj is not None:
                            self.cpp_obj.update_target_position(self.target_pos_received)
                except Exception as e:
                    self.logger_ri.warning("_update_target_info, no position received: %s", e)
                    pass
                time.sleep(self.control_dt / self.check_queue_period_divisor)

        if message == "play_recording":
            # Load recording data
            try:
                filename = full_message["recording_name"]
                filename = os.path.join("data", filename)
                if not os.path.exists(filename):
                    self.logger_ri.error("play_recording, file %s not found.", filename)
                    return False
            except KeyError:
                self.logger_ri.error("_update_target_info, missing recording_name in message.")
                return False

            # Load JSON
            with open(filename, 'r') as f:
                data = json.load(f)
                self.teachbot_positions = deque([s["teachbot_position"] for s in data["samples"]])
                try:
                    self.first_joint_position = data["samples"][0]["robot_position"]
                except IndexError:
                    self.logger_ri.error("_update_target_info, no robot positions found in %s.", filename)
                    return False

            if not data:
                self.logger_ri.error("_update_target_info, %s is empty.", filename)
                return False

            # Check playback_speed
            self.default_recording_speed = data["metadata"]["recording_speed"]
            playback_speed = full_message.get("playback_speed", "")
            if playback_speed == "" or playback_speed is None:
                self.robot_speed = self.default_recording_speed
                self.logger_ri.info(
                    "_update_target_info: Empty or null playback_speed, using default: %s",
                    self.default_recording_speed
                )
            else:
                self.robot_speed = playback_speed * self.default_recording_speed
                self.logger_ri.info("_update_target_info, playback_speed: %s", self.robot_speed)

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
        if not self._move_to_start_position():
            self.logger_ri.error("_control_robot, could not move to start position.")
            return 

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
        # set robot speed at 100% 
        self.set_speed_overwrite(100)

        position = copy.deepcopy(self.start_position)
        if not self.push_joint_motion(position, speed=10, term_type="FINE", term_val=0):
            self.logger_ri.error("_move_to_start_position, could not move to start position.")
            return False
        self.udp.joint_state_received = self.start_position
        self.logger_ri.info("_move_to_start_position, moved to start position.")
        return True

    def _check_start_position(self, full_message):
        """
        Verify that the current position is near the start position,
        or handle 'opening_ceremony' if teleoperation is started.
        """
        message = full_message.get("message", "")

        if message == "start_teleoperation" or message == "start_teleoperation_record" or message == "record_episodes" or message == "record_episode":
            self.logger_ri.info("checking start position")
            if not self.opening_ceremony():
                self.logger_ri.error("_check_start_position, opening ceremony failed")
                return False

        if message == "play_recording":
            # Check if start position is close to the first recording position
            diffs = [abs(c - s) for c, s in zip(self.start_position, self.first_joint_position)]
            if any(d > self.start_joint_tolerance for d in diffs):
                self.logger_ri.error(
                    "_check_start_position, start position is not close to the first position in the recording"
                )
                return False
            self.logger_ri.info("_check_start_position, checking start position")
            return True
        
        if message == "run_policy":
            return True

    def _prepare_robot(self, full_message):
        """
        Make the robot ready for streaming:
        - Call TPP program via RMI
        - Start UDP streaming
        """

        # Determine robot limits
        current_speed = 200
        joint_limits = [self.udp.get_limit_values(i + 1, current_speed) for i in range(self.dof)]
        self.joint_limits = self.udp.get_joint_limits(joint_limits, self.start_position)


        started_streaming = False
        if self.robot_running:
            # Call the TPP program
            if not self.rmi.call_motion_stream():
                self.logger_ri.error("_prepare_robot, could not call motion stream")
                return

            self.logger_ri.info("_prepare_robot, starting streaming")
            self.udp.send_start_packet()
            started_streaming = True

        return started_streaming

    def _control_loop(self, started_streaming):
        """
        Core loop for either teleoperation or playing back a recorded trajectory:
        - Start Receive loop for robot responses
        - Setup Ruckig
        - Fill action buffer
        - While running, compute next motion using Ruckig and send to robot
        - Optionally store motion data if recording
        - Stop receive loop after control loop ends
        """

        if self.control_loop_language == "python":
            # start receive loop for robot responses
            self.logger_ri.info("control_loop, starting receive loop for robot responses")
            self.udp.start_receiving_thread()
            self.logger_ri.info("control_loop, receive loop started")

            # Set up Ruckig with initial conditions
            self.logger_ri.info("control_loop, Setup Ruckig")
            otg, inp, out = self.setup_ruckig()
            self.logger_ri.info("control_loop, Ruckig setup done.")

            # Garbage collection optimization for real-time performance
            # Store original GC settings to restore later
            original_gc_thresholds = gc.get_threshold()
            
            # Set optimized GC thresholds for real-time performance
            # Higher thresholds = less frequent GC = more predictable timing
            optimized_gen0 = 2000   # Less frequent gen0 (default: 700)
            optimized_gen1 = 25     # Less frequent gen1 (default: 10)
            optimized_gen2 = 25     # Less frequent gen2 (default: 10)
            gc.set_threshold(optimized_gen0, optimized_gen1, optimized_gen2)
            
            time.sleep(0.01)

            # Wait for the first status packet
            while not self.udp.started_receiving_motion_stream:
                time.sleep(self.control_dt / 4)

            self.logger_ri.info("control_loop, streaming started, filling buffer")
            for i in range(self.action_buffer_length):
                # Extract gripper state from start_position (last element)
                gripper_state = 1 if (len(self.start_position) > self.dof and self.start_position[-1] > 0.5) else 0
                if i == 0:
                    gripper_state = 1
                self.udp.send_joint_pos(self.start_position[:-1], gripper_state, False)

            self.logger_ri.info("control_loop, starting control loop")
            previous_action_master = self.start_position


            while self.robot_running:
                # Sync with the packets so we don't overrun the buffer
                while self.udp.seq_id_received + self.action_buffer_length < self.udp.seq_id_sent:
                    time.sleep(self.control_dt / self.check_queue_period_divisor / 2)
                
                last_received_time = self.udp.joint_state_received_time
                last_received_js = self.udp.joint_state_received

                # Get next action
                if self.play_recording_active:
                    if self.teachbot_positions:
                        action_teachbot = self.teachbot_positions.popleft()
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
                        break

                # Ruckig trajectory calculation
                current_position = self.update_ruckig_input(action_master, inp, previous_action_master)
                previous_action_master = action_master

                success_calc = self.trajectory_calculation(otg, inp, out, current_position)
                
                self.determine_gripper_state(action_master[-1])
                
                if not success_calc:
                    self.logger_ri.error("control_loop, trajectory calculation failed")
                    break

                sent_robot_position = out.new_position
                teachbot_position = inp.target_position

                # Store flags before sending
                gripper_on_to_send = self.gripper_on
                gripper_off_to_send = self.gripper_off
                success_send = self.udp.send_joint_pos(sent_robot_position, gripper_on_to_send, gripper_off_to_send)
                
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
                        
                        # Convert gripper state to float (0.0 or 1.0) and append to position arrays
                        gripper_state_float = 1.0 if self.gripper_state else 0.0
                        teachbot_pos_with_gripper = teachbot_position + [gripper_state_float]
                        sent_pos_with_gripper = sent_robot_position + [gripper_state_float]
                        # last_received_js already contains gripper state as last element from UDP interface
                        
                        joint_data = {
                            "teachbot_position": teachbot_pos_with_gripper,
                            "sent_robot_position": sent_pos_with_gripper,
                            "robot_position": last_received_js,  # Already includes gripper state
                            "robot_position_timestamp": last_received_time,
                            "seq_id": self.udp.seq_id_sent,
                        }
                        self.shm_joint_data1.put(joint_data)
                    except Exception as e:
                        self.logger_ri.error("control_loop, error uploading joint data: %s", e)

                if self.run_policy_active:
                    try:
                        if self.shm_joint_data2.full():
                            self.shm_joint_data2.get_nowait()
                        
                        # Convert gripper state to float (0.0 or 1.0) and append to position arrays
                        gripper_state_float = 1.0 if self.gripper_state else 0.0
                        teachbot_pos_with_gripper = teachbot_position + [gripper_state_float]
                        sent_pos_with_gripper = sent_robot_position + [gripper_state_float]
                        # last_received_js already contains gripper state as last element from UDP interface
                        
                        joint_data = {
                            "teachbot_position": teachbot_pos_with_gripper,
                            "sent_robot_position": sent_pos_with_gripper,
                            "robot_position": last_received_js,  # Already includes gripper state
                            "robot_position_timestamp": last_received_time,
                            "seq_id": self.udp.seq_id_sent,
                        }
                        self.shm_joint_data2.put(joint_data)
                    except Exception as e:
                        self.logger_ri.error("control_loop, error uploading joint data: %s", e)

            # Restore original GC settings when exiting
            gc.set_threshold(*original_gc_thresholds)

            if started_streaming:
                self.logger_ri.info("control_loop, stopping robot")
                self.stop_robot(otg, inp, out)
                self.logger_ri.info("control_loop, robot stopped")

            # Stop receiving target positions
            self.logger_ri.info("control_loop, stopping receive loop for target positions")
            self.udp.stop_receiving_thread()
            self.logger_ri.info("control_loop, receive loop stopped")
        


        elif self.control_loop_language == "cpp":
            self.robot_cleanup_done = False

            # close the udp socket in python
            self.udp.close_socket()
            self.server_address[1] = str(self.server_address[1])  # Ensure port is a string
            self.robot_address[1] = str(self.robot_address[1])  # Ensure port is a string
            if self.target_pos_received is None:
                self.target_pos_received = self.start_position

            # 3) Create the C++ object with all parameters including shared memory config
            cpp_config = self.config["cpp"]["shared_memory"]
            shm1_config = cpp_config["shm_joint_data1"]
            shm2_config = cpp_config["shm_joint_data2"]
            
            # Calculate capacities using deterministic formula (same as Python uses)
            from utils.utils import calculate_shm_capacity
            # Use record_duration from config for shm_joint_data1 (recording buffer)
            shm_capacity = calculate_shm_capacity(self.record_duration, self.control_dt)
            shm_name = shm1_config["shm_name"]
            
            # Policy buffer uses capacity from config for shm_joint_data2
            shm_policy_capacity = shm2_config["capacity"]
            shm_policy_name = shm2_config["shm_name"]
            
            self.logger_ri.info("C++ shared memory config:")
            self.logger_ri.info("  record_duration: %.2f s, control_dt: %.4f s", self.record_duration, self.control_dt)
            self.logger_ri.info("  shm_capacity (shm1, calculated): %d", shm_capacity)
            self.logger_ri.info("  shm_policy_capacity (shm2, from config): %d", shm_policy_capacity)
            
            # Get policy target shared memory info - set defaults first
            shm_policy_target_capacity = 0
            shm_policy_target_name = ""
            shm_policy_target_format = ""
            shm_policy_target_entry_size = 0
            
            # Override with actual values if policy is active
            if self.run_policy_active:
                shm_policy_target_capacity = self.shm_target_pos2_capacity if self.shm_target_pos2 else 0
                shm_policy_target_name = self.shm_target_pos2_info['name'] if self.shm_target_pos2_info else ""
                shm_policy_target_format = self.shm_target_pos2_entry_format if self.shm_target_pos2 else ""
                shm_policy_target_entry_size = self.shm_target_pos2_entry_size if self.shm_target_pos2 else 0
            
            self.cpp_obj = control_loop_module.ControlLoop(
                #  1) bool started_streaming
                started_streaming,
                #  2) double control_dt
                self.control_dt,
                #  3) int dof
                self.dof,
                #  4) vector<double> start_position
                self.start_position,
                #  5) vector< array<double,3> > joint_limits
                self.joint_limits,  # must be e.g. [[v1,a1,j1], [v2,a2,j2], ...]
                #  6) bool joint_synchronization
                self.joint_synchronization,
                #  7) int action_buffer_length
                self.action_buffer_length,
                #  8) string server_address
                self.server_address,
                #  8) string robot_address
                self.robot_address,
                #  9) int check_queue_period_divisor
                self.check_queue_period_divisor,
                # 10) bool play_recording_active
                self.play_recording_active,
                # 11) deque<vector<double>> teachbot_positions
                self.teachbot_positions,  # e.g., a deque of lists from your code
                # 12) double gripper_treshold
                self.gripper_treshold,
                # 13) double gripper_delay
                self.gripper_delay,
                # 14) double robot_speed
                self.robot_speed,
                # 15) vector<double> target_pos_received
                self.target_pos_received,
                # 16) bool robot_running
                self.robot_running,
                # 17) bool recording
                self.recording,
                # 18) vector<double> upper_limits
                self.upper_limits,
                # 19) vector<double> lower_limits
                self.lower_limits,
                # 20) bool gripper_on (deprecated - now using gripper state in joint arrays)
                False,
                # 21) bool gripper_off (deprecated - now using gripper state in joint arrays)
                False,
                # 22) bool gripper_state
                self.gripper_state,
                # 23) python function
                self.python_callback,
                # 24) int shm_capacity
                shm_capacity,
                # 25) string shm_name  
                shm_name,
                # 26) int shm_policy_capacity
                shm_policy_capacity,
                # 27) string shm_policy_name
                shm_policy_name,
                # 28) int shm_policy_target_capacity
                shm_policy_target_capacity,
                # 29) string shm_policy_target_name
                shm_policy_target_name,
                # 30) string shm_policy_target_format
                shm_policy_target_format,
                # 31) int shm_policy_target_entry_size
                shm_policy_target_entry_size
            )

            self.logger_ri.info("control_loop, C++ object created with %s", self.robot_running)
            
            # Set policy active state in C++ object if policy is running
            if self.run_policy_active:
                self.cpp_obj.set_policy_active(True)
                self.logger_ri.info("control_loop, Policy control activated in C++ control loop")
            
            self.cpp_obj.start_control_loop(started_streaming)
            self.logger_ri.info("control_loop, C++ control loop started")

            # C++ code now handles policy target reading directly from shared memory
            
            while self.robot_running:
                time.sleep(self.control_dt / self.check_queue_period_divisor)
            self.logger_ri.info("control_loop, C++ control loop ending, sending stop command to cpp")

            # Close the cpp object
            self.cpp_obj.stop_control_loop()
            self.logger_ri.info("control_loop, C++ control loop stopped, waiting for cleanup to finish")
            while not self.robot_cleanup_done:
                time.sleep(self.control_dt / self.check_queue_period_divisor)
            self.logger_ri.info("control_loop, C++ cleanup finished")

            # reestablish the UDP socket in Python
            self.logger_ri.info("control_loop, reestablishing UDP socket in Python")
            if not self.udp.setup_udp_socket():
                self.logger_ri.error("control_loop, could not reestablish UDP socket")
                return False

        else:
            self.logger_ri.error(
                "control_loop, unsupported control loop language: %s", self.control_loop_language
            )

    def _close_robot(self, started_streaming):
        """
        Close out any running streaming after control loop ends.
        """
        if started_streaming:
            self.logger_ri.info("_close_robot, stopping UDP streaming")
            self.udp.send_stop_packet()
            self.logger_ri.info("_close_robot, UDP streaming stopped")

    def _send_stop_response(self, full_message):
        """
        If the robot was playing a recording, send a 'stop' command
        back to the queue to indicate the operation completed.
        """
        message = full_message.get("message", "")
        if message == "play_recording":
            send_tc_command(self.robot_interface_commup, {"type": "CMD", "message": "stop", "interface": "ROBOT_INTERFACE"})
            self.logger_ri.info("_send_stop_response, sent stop command to queue.")

    ################################################################
    # 3) ROBOT-SPECIFIC LOGIC (Connection, RMI calls, Ruckig, etc.)
    ################################################################

    def connect(self):
        """
        1) Uses RMIConnection to do the TCP handshake.
        2) Sets up UDP socket.
        3) Resets, initializes RMI.
        4) Creates limit table.
        """
        if self.connected:
            self.logger_ri.info("connect, Already connected.")
            return True

        # STEP 1: Connect RMI
        if not self.rmi.connect():
            self.logger_ri.error("connect, RMI connect failed.")
            self._cleanup_connection()
            return False
        self.rmi_connected = True

        # STEP 2: Set up the UDP socket
        if not self.udp.setup_udp_socket():
            self.logger_ri.error("connect, Could not set up UDP socket.")
            self._cleanup_connection()
            return False
        self.logger_ri.info("connect, UDP socket set up.")

        # STEP 3: RMI reset
        if not self.rmi.reset():
            self.logger_ri.error("connect, RMI reset failed.")
            self._cleanup_connection()
            return False
        self.logger_ri.info("connect, RMI reset.")

        # STEP 4: RMI initialize
        if not self.rmi.initialize_rmi():
            self.rmi.reset()
            self.logger_ri.error("connect, RMI initialize failed.")
            self._cleanup_connection()
            return False
        self.logger_ri.info("connect, RMI initialized.")

        # STEP 5: Fill limit tables
        if not self.udp.create_limit_table(self.payload_weight, self.robot_max_payload):
            self.logger_ri.error("connect, create_limit_table failed.")
            self._cleanup_connection()
            return False
        self.logger_ri.info("connect, limit tables created.")

        self.logger_ri.info("connect, all steps successful. Robot is connected.")
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
        
        # Also update C++ object if using C++ control loop
        if self.control_loop_language == "cpp" and hasattr(self, 'cpp_obj') and self.cpp_obj:
            self.cpp_obj.set_policy_active(False)
            
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

        self.logger_ri.info("_cleanup_connection: All connections closed; ready to reconnect.")

    def push_joint_motion(self, position, speed=10, term_type="FINE", term_val=0):
        """
        Wrapper around RMIConnection.push_joint_motion.
        Applies J3_interaction if configured, then calls the RMI method.
        """
        if self.joint3_interaction:
            position = J3_interaction(position)
        return self.rmi.push_joint_motion(position, speed, term_type, term_val)

    def set_speed_overwrite(self, speed):
        """
        Set robot speed override via RMI connection.
        
        :param speed: Speed override percentage (0-100)
        """
        if not self.rmi_connected:
            self.logger_ri.error("set_speed_overwrite: RMI not connected")
            return False
            
        data = {"Command": "FRC_SetOverRide", "Value": speed}
        try:
            self.rmi._send_json(self.rmi.tcp_socket, data)
            self.logger_ri.info(f"set_speed_overwrite: Set speed override to {speed}%")
            return True
        except Exception as e:
            self.logger_ri.error(f"set_speed_overwrite: Failed to set speed override: {e}")
            return False

    def stop_robot_running(self):
        """
        Stop the teleoperation or play_recording threads and reset relevant flags.
        """
        self.receive_target_pos = False
        self.robot_running = False
        self.gripper_delay = 0

        # Join the receive_target_pos_thread
        if self.update_target_info_thread:
            try:
                self.update_target_info_thread.join(timeout=0.1)
                self.update_target_info_thread = None
                self.logger_ri.info("stop_robot_running, target position listening stopped.")
            except Exception as e:
                self.logger_ri.error("stop_robot_running, error joining receive_target_pos_thread: %s", e)

        # stop control loop if robot is running
        if self.robot_control_thread:
            try:
                self.robot_control_thread.join(timeout=5)
                self.robot_control_thread = None
                self.logger_ri.info("stop_robot_running, robot control thread stopped.")
            except Exception as e:
                self.logger_ri.error("stop_robot_running, error joining robot_control thread: %s", e)

        return True

    def opening_ceremony(self):
        """
        Wait until the externally-provided target is within tolerance of the start position.
        """
        ready_for_teleoperation = False
        self.target_pos_received = None
        time.sleep(0.01)
        while self.robot_running:
            if not ready_for_teleoperation:
                if self.target_pos_received is not None:
                    diffs = [abs(c - s) for c, s in zip(self.target_pos_received, self.start_position)]

                    if all(d < self.start_joint_tolerance for d in diffs):
                        ready_for_teleoperation = True
                        self.logger_ri.info(
                            "teleoperation loop, position within tolerance, starting teleoperation"
                        )
                        return True
                # else:
                #     self.logger_ri.warning("teleoperation loop, no target position received yet")
                time.sleep(self.control_dt)
        return False


    ################################################################
    # 4) Ruckig Pathplanner logic
    ################################################################
    def setup_ruckig(self):
        """
        Set up Ruckig with initial conditions.
        """
        otg = Ruckig(self.dof, self.control_dt)
        inp = InputParameter(self.dof)
        out = OutputParameter(self.dof)

        inp.current_position = self.start_position[:self.dof]
        inp.current_velocity = [0.0] * self.dof
        inp.current_acceleration = [0.0] * self.dof
        inp.target_position = self.start_position[:self.dof]
        inp.target_velocity = [0.0] * self.dof
        inp.target_acceleration = [0.0] * self.dof

        inp.max_velocity = [
            round((limit[0] * self.robot_speed), 6) for limit in self.joint_limits
        ]
        inp.max_acceleration = [
            round((limit[1] * (self.robot_speed ** 2)), 6) for limit in self.joint_limits
        ]
        inp.max_jerk = [
            round(limit[2] * (self.robot_speed ** 3)) for limit in self.joint_limits
        ]

        # Synchronization of the joints
        if not self.joint_synchronization:
            inp.synchronization = Synchronization.No

        return otg, inp, out

    def stop_robot(self, otg, inp, out):
        """
        Smoothly stop the robot by continuing Ruckig until it settles.
        Then send a few more actions to ensure it's stationary.
        """
        from ruckig import Result

        res = Result.Working
        nr_stop_actions = 50
        stop_step_counter = 0

        self.logger_ri.info("stop_robot, reducing speed")

        while res == Result.Working:
            while self.udp.seq_id_received + self.action_buffer_length < self.udp.seq_id_sent:
                time.sleep(self.control_dt / self.check_queue_period_divisor)
            res = otg.update(inp, out)
            out.pass_to_input(inp)
            action_puppet = copy.deepcopy(out.new_position)
            # During stop, don't change gripper state - action_puppet only contains 6 robot joints
            self.udp.send_joint_pos(action_puppet, False, False)

        self.logger_ri.info("stop_robot, stopped robot")
        self.logger_ri.info("stop_robot, sending %s more actions", nr_stop_actions)
        while stop_step_counter < nr_stop_actions:
            while self.udp.seq_id_received + self.action_buffer_length < self.udp.seq_id_sent:
                time.sleep(self.control_dt / self.check_queue_period_divisor)
            # During stop, don't change gripper state - action_puppet only contains 6 robot joints
            stop_step_counter += 1
            self.udp.send_joint_pos(action_puppet, False, False)
        self.logger_ri.info("stop_robot, done stopping robot")
        return True

    def update_ruckig_input(self, action_master, inp, previous_action_master):
        """
        Process the new action (master) against position limits and
        set the Ruckig input accordingly.
        """
        current_position = round_position(inp.current_position)
        inp.current_position = current_position

        for i in range(self.dof):
            if action_master[i] > self.upper_limits[i]:
                action_master[i] = self.upper_limits[i]
            elif action_master[i] < self.lower_limits[i]:
                action_master[i] = self.lower_limits[i]

        # # Stick if tiny difference
        # for i in range(self.dof):
        #     if abs(action_master[i] - current_position[i]) < 0.2:
        #         if abs(action_master[i] - previous_action_master[i]) < 0.2:
        #             action_master[i] = previous_action_master[i]

        inp.target_position = action_master[:self.dof]
        
        return current_position

    def trajectory_calculation(self, otg, inp, out, current_position):
        """
        Perform the Ruckig update step, handling any exceptions in handle_trajectory_error.
        """
        start_time = time.time()
        try:
            otg.update(inp, out)
        except Exception as e:
            result = self.handle_trajectory_error(e, inp, out, otg, self.dof)
            if not result:
                self.logger_ri.error(
                    "trajectory_calculation, stopping robot due to Ruckig error: %s", e
                )
                return False
            else:
                (action_puppet, otg, new_inp, new_out) = result
                inp = new_inp
                out = new_out
                out.pass_to_input(inp)

        out.pass_to_input(inp)
        elapsed = time.time() - start_time
        if elapsed > self.control_dt:
            self.logger_ri.warning(
                "trajectory_calculation, calculation took %.4f s > %.4f s",
                elapsed, self.control_dt
            )
        return True

    def handle_trajectory_error(self, e, inp, out, otg, dof):
        """
        Attempt to correct certain Ruckig errors by adjusting target DOF or velocity/acceleration,
        returning the 'puppet' action if successful; otherwise return False.
        """
        error_msg = str(e)
        match = re.search(r'dof: (\d+)', error_msg)
        if not hasattr(self, 'error_3_occur'):
            self.error_3_occur = 0

        if match:
            try:
                error_dof = int(match.group(1))
                target = inp.target_position
                current_velocity = inp.current_velocity
                current_acceleration = inp.current_acceleration

                for i in range(dof):
                    current_velocity[i] = round(current_velocity[i], 6)
                    current_acceleration[i] = round(current_acceleration[i], 6)

                target[error_dof] += 0.0001
                inp.target_position = target
                inp.current_velocity = current_velocity
                inp.current_acceleration = current_acceleration

                otg.update(inp, out)
                action_puppet = out.new_position
                out.pass_to_input(inp)
                return (action_puppet, otg, inp, out)

            except Exception as e2:
                error2_msg = str(e2)
                match2 = re.search(r'dof: (\d+)', error2_msg)
                if match2:
                    try:
                        error2_dof = int(match2.group(1))
                        target = inp.target_position
                        current_pos = inp.current_position
                        target[error2_dof] = current_pos[error2_dof]
                        inp.target_position = target
                        otg.update(inp, out)
                        self.logger_ri.warning(
                            "Ruckig cannot calculate. Sent DOF %d to current position.", error2_dof
                        )
                        out.pass_to_input(inp)
                        return (out.new_position, otg, inp, out)
                    except Exception as e3:
                        self.logger_ri.warning("Ruckig cannot calculate: %s", e3)
                        self.error_3_occur += 1
                        if self.error_3_occur > 5:
                            return False

                        import numpy as np
                        current_position = np.array(inp.current_position)
                        current_velocity = np.array(inp.current_velocity)
                        current_acceleration = np.array(inp.current_acceleration)
                        max_velocity = np.array(inp.max_velocity)
                        max_acceleration = np.array(inp.max_acceleration)
                        max_jerk = np.array(inp.max_jerk)

                        desired_acc = -np.sign(current_velocity) * max_acceleration
                        delta_acc = np.clip(
                            desired_acc - current_acceleration,
                            -max_jerk * self.control_dt * 0.8,
                            max_jerk * self.control_dt * 0.8
                        )
                        next_acc = current_acceleration + delta_acc
                        next_acc = np.clip(next_acc, -max_acceleration, max_acceleration)
                        next_vel = current_velocity + next_acc * self.control_dt
                        next_vel = np.clip(next_vel, -max_velocity, max_velocity)
                        next_pos = current_position + current_velocity * self.control_dt \
                            + 0.5 * current_acceleration * (self.control_dt ** 2)

                        inp.current_position = next_pos.tolist()
                        inp.current_velocity = next_vel.tolist()
                        inp.current_acceleration = next_acc.tolist()

                        action_puppet = next_pos.tolist()
                        return (action_puppet, otg, inp, out)
                else:
                    return False
        else:
            return False

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
            if self.gripper_delay > 0.0 and not self.run_policy_active:
                self.gripper_state_change_time_threshold = self.gripper_delay / self.robot_speed
                # Check if enough time has passed since last state change
                if (now - self.gripper_state_change_time) >= self.gripper_state_change_time_threshold:
                    self.gripper_on = True
                    self.gripper_state = 1
                    self.gripper_state_change_time = now
            else:
                self.gripper_on = True
                self.gripper_state = 1
                self.gripper_state_change_time = now

        # Turn OFF
        elif gripper_state < self.gripper_treshold and self.gripper_state:
            self.gripper_off = True
            self.gripper_state = 0
            self.gripper_state_change_time = now

    ############################################################################
    # 5) PROCESS PRIORITY ADJUSTMENT AND CPP CALLBACK
    ############################################################################


    def python_callback(self):
        self.logger_ri.info("Python callback called from C++ control loop.")
        self.robot_running = False
        self.robot_cleanup_done = True
        self.logger_ri.info("Python callback called, robot running is false.")
        return






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
        
        # Adjust process priority if configured
        if config["general"].get("robot_interface_sudo_priority", False):
            priority = config["general"].get("robot_interface_priority_value", -10)
            try:
                logger_ri.info(f"which python: {sys.executable}")

                os.setpriority(os.PRIO_PROCESS, 0, priority)
                logger_ri.info(f"Process priority set to {priority} (nice value)")
            except PermissionError:
                logger_ri.warning(f"Permission denied: Cannot set process priority to {priority}. "
                                "Consider running with appropriate privileges or using 'nice' command.")
            except Exception as e:
                logger_ri.error(f"Failed to set process priority: {e}")

        
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
                            
                elif message == "record_episode":
                    if robot.robot_running:
                        logger_ri.warning(
                            "[CMD] record_episode unsuccessful, Robot already running"
                        )
                        send_response(logger_ri, robot_interface_commup, full_message,
                                      error="Robot already running")
                    else:
                        if robot.record_episode(full_message):
                            logger_ri.info("record_episode successful")
                            send_response(logger_ri, robot_interface_commup, full_message, error="None")
                        else:
                            logger_ri.error("record_episode unsuccessful, Could not record episode")
                            send_response(logger_ri, robot_interface_commup, full_message,
                                          error="Could not record episode")


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
