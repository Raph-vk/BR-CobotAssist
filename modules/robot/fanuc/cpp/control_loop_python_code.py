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

from modules.robot_0interface import RobotInterface
from ruckig import InputParameter, OutputParameter, Ruckig, Result, Synchronization

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
from utils.utils import setup_logging, load_config
from ..robot_fanuc_rmi import RMIConnection
from ..robot_fanuc_udp import UDPStreaming


class control_loop():

    def __init__(self):

        # variables 
        self.server_adress
        self.udp_socket # Bind local UDP socket to self.server_address, set timeouts, etc.
        self.seq_id_received = 0 # 1 time passed at the beginning
        self.joint_state_received = None # 1 time passed at the beginning
        self.started_receiving_motion_stream = False # 1 time passed at the beginning

        self.started_streaming # 1 time passed at the beginning
        self.control_dt, # 1 time passed at the beginning
        self.dof, # 1 time passed at the beginning
        self.start_position, # 1 time passed at the beginning
        self.joint_limits, # 1 time passed at the beginning
        self.joint_synchronization, # 1 time passed at the beginning
        self.action_buffer_length, # 1 time passed at the beginning
        self.robot_address, # 1 time passed at the beginning
        self.check_queue_period_divisor, # 1 time passed at the beginning
        self.play_recording_active, # 1 time passed at the beginning
        self.master_positions,  # 1 time passed at the beginning
        self.gripper_treshold, # 1 time passed at the beginning
        self.gripper_state_change_time_treshold, # 1 time passed at the beginning
        self.gripper_delay, # 1 time passed at the beginning
        self.robot_speed, # 1 time passed at the beginning
        self.target_pos_received, # 1 time passed at the beginning
        self.robot_running, # Python updates this variable via a c++ function call to toggle when robot needs to stop, keep purely in c++
        self.recording, # Python updates this variable via a c++ function call to toggle when robot needs to start/stop recording, keep purely in c++
        self.shm_joint_data, # This will be replaced by a shared memory object and passed from python to c++, store as global or class member, and added to shm by aquiring GIL, construct dict and call queue.put(dict) using python api
        self.upper_limits, # 1 time passed at the beginning
        self.lower_limits, # 1 time passed at the beginning
        self.gripper_on, # 1 time passed at the beginning
        self.gripper_off, # 1 time passed at the beginning
        self.gripper_state, # 1 time passed at the beginning



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
        # start receive loop for robot responses
        self.logger_ri.info("control_loop, starting receive loop for robot responses")
        self.start_receiving()
        self.logger_ri.info("control_loop, receive loop started")

        # Set up Ruckig with initial conditions
        self.logger_ri.info("control_loop, Setup Ruckig")
        otg, inp, out = self.setup_ruckig()
        self.logger_ri.info("control_loop, Ruckig setup done.")

        # Wait for the first status packet
        while not self.started_receiving_motion_stream:
            time.sleep(self.control_dt / 4)

        self.logger_ri.info("control_loop, streaming started, filling buffer")
        for _ in range(self.action_buffer_length):
            self.send_joint_pos(self.start_position, self.gripper_on, self.gripper_off)

        self.logger_ri.info("control_loop, starting control loop")
        previous_action_master = self.start_position

        while self.robot_running:
            # Sync with the packets so we don't overrun the buffer
            while self.seq_id_received + self.action_buffer_length < self.seq_id_sent:
                time.sleep(self.control_dt / self.check_queue_period_divisor / 2)

            # If play_recording is active, read from self.master_positions
            if self.play_recording_active:
                if self.master_positions:
                    action_master = self.master_positions.popleft()
                else:
                    self.logger_ri.info("control_loop, recording playback completed")
                    break
            else:
                if self.target_pos_received is not None:
                    action_master = self.target_pos_received
                else:
                    break

            # Ruckig input updates
            current_position = self.update_ruckig_input(action_master, inp, previous_action_master)
            previous_action_master = action_master

            success_calc = self.trajectory_calculation(otg, inp, out, current_position)
            self.determine_gripper_state(action_master[-1])
            if not success_calc:
                self.logger_ri.error("control_loop, trajectory calculation failed")
                break

            robot_position = out.new_position
            master_position = inp.target_position

            success_send = self.send_joint_pos(robot_position, self.gripper_on, self.gripper_off)
            if self.gripper_on:
                self.gripper_on = False
            if self.gripper_off:
                self.gripper_off = False

            if not success_send:
                self.logger_ri.error("control_loop, could not send joint position to robot")
                break

            # Upload data to shm_joint_data if recording
            if self.recording:
                try:
                    joint_data = {
                        "robot_position": robot_position,
                        "master_position": master_position,
                        "gripper_on": self.gripper_state,
                        "gripper_off": not self.gripper_state
                    }
                    if self.shm_joint_data.full(): # this function will be replaced by a python api call to the shared memory object
                        self.shm_joint_data.get_nowait() # this function will be replaced by a python api call to the shared memory object
                    self.shm_joint_data.put(joint_data) # this function will be replaced by a python api call to the shared memory object
                except Exception as e:
                    self.logger_ri.error("control_loop, error uploading joint data: %s", e)

            # Update last-known robot state
            self.joint_state_received = copy.deepcopy(self.joint_state_received)
            self.joint_state_received_time = self.joint_state_received_time

        if started_streaming:
            self.logger_ri.info("control_loop, stopping robot")
            self.stop_robot(otg, inp, out)
            self.logger_ri.info("control_loop, robot stopped")

        # Stop receiving target positions
        self.logger_ri.info("control_loop, stopping receive loop for target positions")
        self.stop_receiving()
        self.logger_ri.info("control_loop, receive loop stopped")


    def start_receiving(self):
        self.udp_running = True
        self.udp_thread = threading.Thread(target=self.receive_loop, daemon=True) # no thread necessary in c++, leave it out
        self.udp_thread.start()

    def stop_receiving(self):
        if self.udp_running:
            self.udp_running = False
            if self.udp_thread is not None:
                self.udp_thread.join()
                self.udp_thread = None

    def receive_loop(self):
        """
        Thread that receives status data from the robot.
        """
        while self.udp_running:
            try:
                data, address = self.udp_socket.recvfrom(4096)
            except socket.timeout:
                self.logger.warning("UDP recv timeout.")
                continue
            except Exception as e:
                self.logger.error(f"UDP recv error => {e}")
                continue

            if data:
                self.started_receiving_motion_stream = True
                self._decode_status_packet(data)

    def _decode_status_packet(self, data: bytes):
        """
        Unpack the status packet from the robot. Update self.seq_id_received, joint positions, etc.
        """
        packet_format = '>IIIBBHHHI' + 'f' * 27
        unpacked_data = struct.unpack(packet_format, data)

        (packet_type, version_no, self.seq_id_received, status,
         io_type, io_index, io_mask, io_value, timestamp,
         x, y, z, w, p, r,
         ext1, ext2, ext3,
         j1, j2, j3, j4, j5, j6, j7, j8, j9,
         mc1, mc2, mc3, mc4, mc5, mc6, mc7, mc8, mc9) = unpacked_data

        # For this robot, io_value==2 means gripper is ON => store as 1
        if io_value == 2:
            io_value = 1

        new_joint_state = [j1, j2, j3, j4, j5, j6, io_value]
        new_joint_state = self._J3_interaction_rev(new_joint_state)

        self.joint_state_received = copy.deepcopy(new_joint_state)
        self.joint_state_received_time = time.time()

    def _J3_interaction_rev(self, action):
        """ Same logic as global J3_interaction_rev, but local to UDP class. """
        action[2] = action[2] + action[1]
        return action
    
    def setup_ruckig(self):
        """
        Set up Ruckig with initial conditions.
        """
        otg = Ruckig(self.dof, self.control_dt)
        inp = InputParameter(self.dof)
        out = OutputParameter(self.dof)

        inp.current_position = self.start_position
        inp.current_velocity = [0.0] * self.dof
        inp.current_acceleration = [0.0] * self.dof
        inp.target_position = self.start_position
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
    
    def send_joint_pos(self, position, gripper_on, gripper_off):
        """
        Build and send a packet to command a joint position. Also handle gripper signals.
        """
        # If first send => set seq_id_sent to one less than received
        if self.seq_id_sent == 0:
            self.seq_id_sent = self.seq_id_received - 1

        packet_type = 1
        version_no = 1
        self.seq_id_sent += 1
        last_data = 0
        read_io_type = 9
        read_io_index = 1
        read_io_mask = 0x0002
        data_format = 1  # joint format

        # Evaluate gripper commands
        if gripper_on:
            writing_io_type = 9  # RO type
            writing_io_index = 1  # RO[2]
            writing_io_mask = 0x0002
            writing_io_value = 0x0002  # Turn it ON
        elif gripper_off:
            writing_io_type = 9
            writing_io_index = 1
            writing_io_mask = 0x0002
            writing_io_value = 0x0000  # Turn it OFF
        else:
            writing_io_type = 0
            writing_io_index = 0
            writing_io_mask = 0x0
            writing_io_value = 0x0

        # Pad the position with zeros to length 9
        position = np.pad(position, (0, 9 - len(position)), 'constant')
        position = self._J3_interaction(position)

        packet = struct.pack(
            '>IIIBBHHBBHHHxxfffffffff',
            packet_type, version_no, self.seq_id_sent, last_data,
            read_io_type, read_io_index, read_io_mask, data_format,
            writing_io_type, writing_io_index, writing_io_mask, writing_io_value, *position
        )

        try:
            self.udp_socket.sendto(packet, self.robot_address)
            timenow = time.time()
            # Check how quickly we are sending
            if 0.008 < (timenow - self.time_last_sent) < 747059116:
                self.logger.warning(
                    "send_joint_pos, packet=%d, on_time=%s, time=%.4f",
                    self.seq_id_sent, False, timenow - self.time_last_sent
                )
            self.time_last_sent = timenow
        except Exception as e:
            self.logger.error("send_joint_pos, error sending packet to robot: %s", e)
            return False

        return True


    def _J3_interaction(self, action):
        """ Same logic as global J3_interaction, but local to UDP class. """
        action[2] = action[2] - action[1]
        return action
    
    def update_ruckig_input(self, action_master, inp, previous_action_master):
        """
        Process the new action (master) against position limits and
        set the Ruckig input accordingly.
        """
        current_position = self.round_position(inp.current_position)
        inp.current_position = current_position

        for i in range(self.dof):
            if action_master[i] > self.upper_limits[i]:
                action_master[i] = self.upper_limits[i]
            elif action_master[i] < self.lower_limits[i]:
                action_master[i] = self.lower_limits[i]

        # Stick if tiny difference
        for i in range(self.dof):
            if abs(action_master[i] - current_position[i]) < 0.2:
                if abs(action_master[i] - previous_action_master[i]) < 0.2:
                    action_master[i] = previous_action_master[i]

        inp.target_position = action_master
        return current_position

    def round_position(position):
        """
        Round all values in a position list to 6 decimals.
        """
        return [round(pos, 6) for pos in position]
    
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
        internal 'gripper_delay' logic.
        """
        # Turn ON
        if gripper_state > self.gripper_treshold and not self.gripper_state:
            if self.gripper_delay is True:
                self.gripper_state_change_time_threshold = self.gripper_delay / self.robot_speed
                if self.gripper_state_change_time + self.gripper_state_change_time_threshold > time.time():
                    self.gripper_on = True
                else:
                    self.gripper_on = False
            else:
                self.gripper_on = True
            self.gripper_state = True
            self.gripper_state_change_time = time.time()

        # Turn OFF
        elif gripper_state < self.gripper_treshold and self.gripper_state:
            self.gripper_off = True
            self.gripper_state = False
            self.gripper_state_change_time = time.time()

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
            while self.seq_id_received + self.action_buffer_length < self.seq_id_sent:
                time.sleep(self.control_dt / self.check_queue_period_divisor)
            res = otg.update(inp, out)
            out.pass_to_input(inp)
            action_puppet = copy.deepcopy(out.new_position)
            self.send_joint_pos(action_puppet, self.gripper_on, self.gripper_off)

        self.logger_ri.info("stop_robot, stopped robot")
        self.logger_ri.info("stop_robot, sending %s more actions", nr_stop_actions)
        while stop_step_counter < nr_stop_actions:
            while self.seq_id_received + self.action_buffer_length < self.seq_id_sent:
                time.sleep(self.control_dt / self.check_queue_period_divisor)
            if self.gripper_on:
                self.gripper_off = True
            stop_step_counter += 1
            self.send_joint_pos(action_puppet, self.gripper_on, self.gripper_off)
        self.logger_ri.info("stop_robot, done stopping robot")
        return True
    

