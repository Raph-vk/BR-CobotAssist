
import json
import socket
import time
import re
import os


class RMIConnection:
    """
    Handles all RMI (TCP) communication with the FANUC robot:
      - Connecting to port 16001, then to the returned port.
      - reset(), initialize_rmi(), get_status(), call_motion_stream(), push_joint_motion(), etc.
      - Low-level JSON send/recv methods.
    """

    def __init__(self, robot_address, rmi_port, logger, config):
        """
        :param robot_address: (ip, port) tuple of the robot. The IP is used for RMI connection.
        :param rmi_port: usually 16001.
        :param logger: logger for logging messages.
        :param config: the entire config dictionary.
        """
        self.logger = logger
        self.config = config

        # Robot address (IP, port). We only need the IP for RMI (since we first connect to rmi_port).
        self.robot_address = robot_address[0]
        self.rmi_port = rmi_port

        # Internal state
        self.tcp_socket = None
        self.rmi_connected = False
        self.rmi_seq_id = 0

    ######################################################
    # Commands
    ######################################################
    def connect(self):
        """
        1) Connect to port `rmi_port` (commonly 16001).
        2) Request a new port from the robot (via {"Communication":"FRC_Connect"}).
        3) Close the first socket and open a new TCP socket on the returned port.
        Returns True if successful, otherwise False.
        """
        if self.rmi_connected:
            self.logger.info("RMIConnection.connect: Already connected.")
            return True

        # -- Step 1: Connect to port 16001 (or self.rmi_port) --
        try:
            self.logger.info(
                "RMIConnection.connect: Attempting to connect to %s:%d",
                self.robot_address, self.rmi_port
            )
            sock_16001 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock_16001.settimeout(2.0)
            sock_16001.connect((self.robot_address, self.rmi_port))
        except Exception as e:
            self.logger.error(
                "RMIConnection.connect: Could not connect to %s:%d => %s. "
                "Check IP address, connection and robot ON/OFF state.",
                self.robot_address, self.rmi_port, e
            )
            return False

        # -- Send {"Communication":"FRC_Connect"} and interpret response --
        msg = {"Communication": "FRC_Connect"}
        self._send_json(sock_16001, msg)
        resp = self._recv_json(sock_16001, timeout=3.0)
        if resp is False:
            self.logger.error("RMIConnection.connect: No response from RMI port %d.", self.rmi_port)
            sock_16001.close()
            return False

        err_id = resp.get("ErrorID", -1)
        port_num = resp.get("PortNumber", None)
        if err_id != 0 or not port_num:
            self.logger.error(
                "RMIConnection.connect: ErrorID=%d, PortNumber=%s. Could not establish RMI session.",
                err_id, port_num
            )
            sock_16001.close()
            return False

        # -- Close the initial socket on rmi_port, open a new one on the returned port --
        sock_16001.close()
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.tcp_socket.settimeout(2.0)
            self.tcp_socket.connect((self.robot_address, port_num))
            self.rmi_connected = True
            self.logger.info(
                "RMIConnection.connect: Connected to %s:%d (RMI)",
                self.robot_address, port_num
            )
            return True
        except Exception as e:
            self.logger.error(
                "RMIConnection.connect: Could not connect to %s:%d => %s.",
                self.robot_address, port_num, e
            )
            return False

    def disconnect(self):
        """
        Send FRC_Disconnect to end the RMI session and close the TCP socket.
        """
        if not self.rmi_connected or not self.tcp_socket:
            self.logger.info("RMIConnection.disconnect: Not connected to RMI.")
            return

        try:
            msg = {"Communication": "FRC_Disconnect"}
            self._send_json(self.tcp_socket, msg)
            resp = self._recv_json(self.tcp_socket, timeout=3.0)
            self.logger.info("RMIConnection.disconnect: Received response: %s", resp)
            if resp is False:
                self.logger.error("RMIConnection.disconnect: No response from FRC_Disconnect.")
        except Exception as e:
            self.logger.error("RMIConnection.disconnect: Error during RMI disconnect: %s", e)
        finally:
            if self.tcp_socket:
                self.tcp_socket.close()
                self.tcp_socket = None

        self.rmi_connected = False
        self.logger.info("RMIConnection.disconnect: RMI disconnected.")

    def call_motion_stream(self):
        """
        Example to call the TPP program 'MOTION_STREAM' from RMI.
        """
        self.rmi_seq_id += 1
        data = {
            "Instruction": "FRC_Call",
            "SequenceID": self.rmi_seq_id,
            "ProgramName": "MOTION_STREAM"
        }
        self._send_json(self.tcp_socket, data)
        self.logger.info("RMIConnection.call_motion_stream: Sent FRC_Call for MOTION_STREAM.")
        return True

    def reset(self):
        """
        FRC_Reset: resets robot alarms. Return True if success.
        """
        if not self.rmi_connected or not self.tcp_socket:
            self.logger.error("RMIConnection.reset: Not connected to RMI.")
            return False

        msg = {"Command": "FRC_Reset"}
        self.logger.info("RMIConnection.reset: Sending FRC_Reset.")
        self._send_json(self.tcp_socket, msg)
        resp = self._recv_json(self.tcp_socket)
        self.logger.info("RMIConnection.reset: Received response: %s", resp)
        if not resp:
            return False

        # Sometimes respond with "FRC_SystemFault" first, then another response
        comm = resp.get("Communication", None)
        if comm == 'FRC_SystemFault':
            self.logger.warning("RMIConnection.reset: Received FRC_SystemFault. Reading next response.")
            resp = self._recv_json(self.tcp_socket)
            self.logger.info("RMIConnection.reset: Next response: %s", resp)
    
        # if the recv_json timeout, retry once
        if resp is False:
            self.logger.error("RMIConnection.reset: No response from FRC_Reset.")
            msg = {"Command": "FRC_Reset"}
            self.logger.info("RMIConnection.reset: Resending FRC_Reset.")
            self._send_json(self.tcp_socket, msg)
            resp = self._recv_json(self.tcp_socket)
            self.logger.info("RMIConnection.reset: Received response: %s", resp)

        err_id = resp.get("ErrorID", -1)
        if err_id != 0:
            if err_id in [2556940, 2556943]:
                self.logger.error(
                    "RMIConnection.reset: ErrorID=%d, check if the robot is ON, OFF-mode, in AUTO, etc.",
                    err_id
                )
            else:
                self.logger.error("RMIConnection.reset: ErrorID=%d", err_id)

        return True

    def initialize_rmi(self):
        """
        FRC_Initialize: create RMI_MOVE program if not running. True if success.
        """
        if not self.rmi_connected or not self.tcp_socket:
            self.logger.error("RMIConnection.initialize_rmi: Not connected.")
            return False

        msg = {"Command": "FRC_Initialize"}
        self._send_json(self.tcp_socket, msg)
        resp = self._recv_json(self.tcp_socket)
        if not resp:
            self.logger.error("RMIConnection.initialize_rmi: No response from FRC_Initialize.")
            return False

        err_id = resp.get("ErrorID", -1)
        if err_id != 0:
            self.logger.error("RMIConnection.initialize_rmi: ErrorID=%d", err_id)
            if err_id in [2556943, 2556940]:
                self.logger.error("RMIConnection.initialize_rmi: Check robot ON, OFF-mode, AUTO, etc.")
            return False

        return True

    def push_joint_motion(self, position, speed=10, term_type="FINE", term_val=0):
        """
        Example of FRC_JointMotionJRep.
        Expects 'position' (already processed, e.g. J3_interaction, etc.).
        """
        self.rmi_seq_id += 1
        data = {
            "Instruction": "FRC_JointMotionJRep",
            "SequenceID": self.rmi_seq_id,
            "JointAngle": {
                "J1": position[0],
                "J2": position[1],
                "J3": position[2],
                "J4": position[3],
                "J5": position[4],
                "J6": position[5],
            },
            "SpeedType": "Percent",
            "Speed": speed,
            "TermType": term_type,
            "TermValue": term_val,
            "ACC": 100,
            "NoBlend": "ON"
        }
        nr_try = 10
        while nr_try > 0:
            self._send_json(self.tcp_socket, data)
            resp = self._recv_json(self.tcp_socket)
            if not resp:
                self.logger.error("RMIConnection.push_joint_motion: No response (timeout).")
                return False
            self.logger.info(
                "RMIConnection.push_joint_motion: Received response: %s, instr=%s, seqID=%s",
                resp, resp.get("Instruction"), resp.get("SequenceID")
            )
            if resp.get("Instruction") == "FRC_JointMotionJRep":
                self.logger.info("RMIConnection.push_joint_motion: Confirmed FRC_JointMotionJRep response.")
                break
            nr_try -= 1

        err_id = resp.get("ErrorID", -1)
        if err_id != 0:
            self.logger.error("RMIConnection.push_joint_motion: ErrorID=%d", err_id)
            return False

        self.logger.info("RMIConnection.push_joint_motion: Moved to %s", position)
        return True

    def get_status(self) -> dict:
        """
        FRC_GetStatus: returns dict with controller info (ServoReady, TPMode, etc.).
        """
        if not self.rmi_connected or not self.tcp_socket:
            self.logger.error("RMIConnection.get_status: Not connected.")
            return {}

        msg = {"Command": "FRC_GetStatus"}
        self._send_json(self.tcp_socket, msg)
        resp = self._recv_json(self.tcp_socket)
        if not resp:
            self.logger.error("RMIConnection.get_status: No response.")
            return {}

        self.logger.info("RMIConnection.get_status: Received response: %s", resp)
        return resp

    ######################################################
    # JSON send/recv helpers
    ######################################################
    def _send_json(self, sock: socket.socket, data: dict):
        """
        Send JSON data followed by '\r\n'.
        """
        txt = json.dumps(data) + "\r\n"
        sock.sendall(txt.encode("utf-8"))

    def _recv_json(self, sock: socket.socket, buf_size=2048, timeout=3.0):
        """
        Receive JSON data (first JSON object found).
        Return False on timeout or invalid parse.
        """
        sock.settimeout(timeout)
        try:
            raw = sock.recv(buf_size).decode("utf-8", errors="ignore")
        except socket.timeout:
            self.logger.warning("RMIConnection._recv_json: Timeout.")
            return False
        if not raw:
            self.logger.warning("RMIConnection._recv_json: No data received.")
            return False

        objs = []
        depth = 0
        start = 0
        for i, ch in enumerate(raw):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    chunk = raw[start:i+1]
                    try:
                        obj = json.loads(chunk)
                        objs.append(obj)
                    except json.JSONDecodeError:
                        pass

        if len(objs) == 0:
            self.logger.warning("RMIConnection._recv_json: No valid JSON object found.")
            return False
        return objs[0]
