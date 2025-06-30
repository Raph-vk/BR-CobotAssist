
import time
import numpy as np
import threading
import socket
import struct
from collections import deque
import os
import copy




##################################################################
# UDPStreaming
# Encapsulates all UDP-related logic, including:
#   - Socket setup/teardown
#   - Sending and receiving motion/status packets
#   - Decoding status data
#   - Limit table requests, storage, and interpolation
##################################################################
class UDPStreaming:
    """
    Handles all UDP socket communication with the Fanuc robot:
      - Setting up the UDP socket,
      - Starting/stopping status streaming,
      - Receiving and decoding status packets,
      - Maintaining seq_id_received/seq_id_sent,
      - Handling limit-table requests and interpolation,
      - Sending joint position commands (including gripper).
    """
    def __init__(self, logger, config, server_address, robot_address, control_dt, check_queue_period_divisor):
        self.logger = logger
        self.config = config
        self.server_address = tuple(server_address)
        self.robot_address = tuple(robot_address)
        self.control_dt = control_dt
        self.check_queue_period_divisor = check_queue_period_divisor

        # Socket-related
        self.udp_socket = None
        self.udp_running = False
        self.udp_thread = None
        self.start_packet_sent = False

        # Sequence IDs
        self.seq_id_received = 0
        self.seq_id_sent = 0
        self.started_receiving_motion_stream = False

        # Joint/Gripper data from robot
        self.joint_state_received = None
        self.joint_state_received_time = time.time()

        # Timing debug
        self.time_last_sent = 0

        # For limit tables
        self.velocity_factors = self.config["hardware"]["robot"]["velocity_factors"]
        self.acceleration_factors = self.config["hardware"]["robot"]["acceleration_factors"]
        self.jerk_factors = self.config["hardware"]["robot"]["jerk_factors"]
        self.limit_tables = {}

        # # Store an acceleration/velocity/jerk history if needed
        # self.joint_position_history = deque([[0] * 7] * 4, maxlen=4)
        # self.joint_velocity_history = deque([[0] * 7] * 4, maxlen=4)
        # self.joint_acceleration_history = deque([[0] * 7] * 4, maxlen=4)
        # self.joint_jerk_history = deque([[0] * 7] * 4, maxlen=4)

    ############################################################
    # UDP Socket Setup & Teardown
    ############################################################
    def setup_udp_socket(self):
        """
        Bind local UDP socket to self.server_address, set timeouts, etc.
        """
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.udp_socket.bind(self.server_address)
            self.udp_socket.settimeout(1.0)
            return True
        except Exception as e:
            self.logger.error(f"setup_udp_socket, Unable to bind UDP socket => {e}")
            return False

    def close_socket(self):
        """
        Close UDP socket if open.
        """
        if self.udp_socket:
            self.udp_socket.close()
            self.udp_socket = None

    ############################################################
    # START / STOP STATUS STREAMING
    ############################################################
    def send_start_packet(self):
        """
        1) Send 'status output start packet' to the robot
        2) Start a thread that listens for status packets
        """
        self._send_status_output_start_packet()
        self.start_packet_sent = True

    def send_stop_packet(self):
        """
        1) Set self.udp_running = False
        2) Join thread
        3) Send 'status output stop packet'
        """
        if self.start_packet_sent:
            self._send_status_output_stop_packet()
            self.start_packet_sent = False
            self.logger.info("UDP streaming stopped.")

    def start_receiving_thread(self):
        self.udp_running = True
        self.udp_thread = threading.Thread(target=self.receive_loop, daemon=True)
        self.udp_thread.start()

    def stop_receiving_thread(self):
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

    ############################################################
    # PACKET SEND/RECEIVE HELPERS
    ############################################################
    def _send_status_output_start_packet(self):
        """
        Packet type=0, version=1 -> Start
        """
        packet_type = 0
        version_no = 1
        packet = struct.pack('>II', packet_type, version_no)
        try:
            self.udp_socket.sendto(packet, self.robot_address)
            self.logger.info(
                "Status output start packet sent to %s", self.robot_address
            )
        except Exception as e:
            self.logger.error("Could not send start packet => %s", e)

    def _send_status_output_stop_packet(self):
        """
        Packet type=2, version=1 -> Stop
        """
        packet_type = 2
        version_no = 1
        packet = struct.pack('>II', packet_type, version_no)
        if self.udp_socket:
            try:
                self.udp_socket.sendto(packet, self.robot_address)
                self.logger.info(
                    "Status output stop packet sent to %s, seq_id=%d",
                    self.robot_address, self.seq_id_received
                )
            except Exception as e:
                self.logger.error("Could not send stop packet => %s", e)

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

    # def _update_joint_info(self):
    #     """
    #     (Optional) Update velocity, acceleration, jerk from the new joint states.
    #     Not currently invoked in the main loop, but kept for completeness.
    #     """
    #     self.joint_position_history.append(list(self.joint_state_received))

    #     # Velocity
    #     new_vel = [
    #         (p2 - p1) / self.control_dt
    #         for p1, p2 in zip(
    #             self.joint_position_history[-2], self.joint_position_history[-1]
    #         )
    #     ]
    #     self.joint_velocity_history.append(new_vel)

    #     # Acceleration
    #     new_acc = [
    #         (v2 - v1) / self.control_dt
    #         for v1, v2 in zip(
    #             self.joint_velocity_history[-2], self.joint_velocity_history[-1]
    #         )
    #     ]
    #     self.joint_acceleration_history.append(new_acc)

    #     # Jerk
    #     new_jerk = [
    #         (a2 - a1) / self.control_dt
    #         for a1, a2 in zip(
    #             self.joint_acceleration_history[-2], self.joint_acceleration_history[-1]
    #         )
    #     ]
    #     self.joint_jerk_history.append(new_jerk)

    ############################################################
    # LIMIT TABLE METHODS
    ############################################################
    def create_limit_table(self, payload_weight, max_payload_weight):
        """
        Request limit data from the robot for velocity/accel/jerk,
        store in self.limit_tables (done via UDP packets).
        """
        if not self.udp_socket:
            self.logger.error("create_limit_table, UDP socket not set up.")
            return False
        if payload_weight > max_payload_weight:
            self.logger.error("create_limit_table, payload_weight > max_payload_weight.")
            return False

        for axis_no in range(1, 7):
            self.limit_tables[axis_no] = {}
            for type_limit in ['velocity', 'acceleration', 'jerk']:
                # 1) request
                self._request_limit_table(axis_no, type_limit)
                # 2) receive
                limit_data = self._receive_limit_table()
                if limit_data is False:
                    self.logger.error(
                        "create_limit_table, Could not receive limit data for axis=%d, type=%s. Retrying...",
                        axis_no, type_limit
                    )
                    self._request_limit_table(axis_no, type_limit)
                    limit_data = self._receive_limit_table()
                    if limit_data is False:
                        self.logger.error(
                            "create_limit_table, Could not receive limit data for axis=%d, type=%s. Giving up.",
                            axis_no, type_limit
                        )
                        return False

                no_payload_limits = limit_data[6:26]
                max_payload_limits = limit_data[26:46]

                factor = 1
                if type_limit == 'velocity':
                    factor = self.velocity_factors[axis_no]
                elif type_limit == 'acceleration':
                    factor = self.acceleration_factors[axis_no]
                elif type_limit == 'jerk':
                    factor = self.jerk_factors[axis_no]

                interpolated_limits = [
                    (no_pl - (no_pl - max_pl) * (payload_weight / max_payload_weight)) * factor
                    for no_pl, max_pl in zip(no_payload_limits, max_payload_limits)
                ]
                self.limit_tables[axis_no][type_limit] = {
                    'interpolated': interpolated_limits,
                    'max_speed': limit_data[4]
                }
        return True

    def get_limit_values(self, axis_no, current_speed):
        """
        Interpolate velocity/acc/jerk for a given axis at current_speed.
        """
        if axis_no not in self.limit_tables:
            return [0, 0, 0]  # no data

        types = ['velocity', 'acceleration', 'jerk']
        results = []
        max_speed = self.limit_tables[axis_no]['acceleration']['max_speed']
        speed_threshold = max_speed / 20
        relative_position = (current_speed / max_speed) * 19
        lower_index = int(relative_position)
        upper_index = min(lower_index + 1, 19)
        weight = relative_position - lower_index

        for type_limit in types:
            values = self.limit_tables[axis_no][type_limit]['interpolated']

            if current_speed < speed_threshold:
                # If below threshold => pick the first table value
                idx_threshold = 0
                value_at_thr = values[idx_threshold]
                results.append(value_at_thr)
            elif current_speed <= max_speed:
                # Inside normal range => linear interpolation
                lv = values[lower_index]
                uv = values[upper_index]
                interpolated_val = lv + weight * (uv - lv)
                results.append(interpolated_val)
            else:
                # Above max => extrapolate
                last_idx = len(values) - 1
                if lower_index >= last_idx:
                    lower_index = max(0, last_idx - 1)
                    upper_index = last_idx
                elif upper_index > last_idx:
                    upper_index = last_idx

                lv = values[lower_index]
                uv = values[upper_index]
                if upper_index - lower_index == 0:
                    weight = 0
                else:
                    weight = (current_speed - max_speed) / (max_speed * (upper_index - lower_index) / 19)
                ext_val = lv + weight * (uv - lv)
                results.append(ext_val)

        return results

    def get_joint_limits(self, joint_limits, current_position):
        """
        Apply velocity/acc/jerk factors to each axis.
        """
        for i in range(len(joint_limits)):
            j_idx = i + 1  # axis in [1..6]
            joint_limits[i][0] *= self.velocity_factors[j_idx]
            joint_limits[i][1] *= self.acceleration_factors[j_idx]
            joint_limits[i][2] *= self.jerk_factors[j_idx]
        return joint_limits

    def _request_limit_table(self, axis_no, type_limit):
        """
        Packet type=3, version=1 => request limit table
        type_limit => velocity=0, acceleration=1, jerk=2
        """
        packet_type = 3
        version_no = 1
        map_ = {'velocity': 0, 'acceleration': 1, 'jerk': 2}
        val = map_[type_limit]

        packet = struct.pack('>IIII', packet_type, version_no, axis_no, val)
        if self.udp_socket:
            self.udp_socket.sendto(packet, self.robot_address)

    def _receive_limit_table(self):
        """
        Wait for the limit table response. Usually 184 bytes.
        If fails => return False
        """
        expected_length = 184
        packet_format = '>IIIIII' + 'f' * 40
        start_time = time.time()
        while True:
            if time.time() - start_time > 2:
                self.logger.warning("Timeout receiving limit table data.")
                return False
            try:
                data, addr = self.udp_socket.recvfrom(4096)
            except socket.timeout:
                continue
            if len(data) >= expected_length:
                unpacked = struct.unpack(packet_format, data)
                return unpacked

    ############################################################
    # SENDING JOINT POSITIONS
    ############################################################
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

    ############################################################
    # MISC HELPERS
    ############################################################
    def _J3_interaction(self, action):
        """ Same logic as global J3_interaction, but local to UDP class. """
        action[2] = action[2] - action[1]
        return action

    def _J3_interaction_rev(self, action):
        """ Same logic as global J3_interaction_rev, but local to UDP class. """
        action[2] = action[2] + action[1]
        return action

    def adjust_process_priority(self, priority):
        """
        Adjust current process priority (requires sudo privileges).
        """
        pid = os.getpid()
        self.logger.info(
            "UDPStreaming.adjust_process_priority, pid=%d, priority=%d. "
            "If sudo is needed, add to /etc/sudoers: "
            "username ALL=(ALL) NOPASSWD: /usr/bin/renice",
            pid, priority
        )
        os.system(f'sudo renice -n {priority} -p {pid}')
