# my_app/modules/teachbot_dummy.py


import time
import copy
import numpy as np
import threading
import subprocess
import os
import sys
import signal
from threading import Thread

from .teachbot_0interface import TeachbotInterface

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.utils import setup_logging, load_config


def send_response(logger_ti, teachbot_interface_commup, payload, error="None", **kwargs):
    """
    Helper to construct and send the new response dict:
        {
        "type": "RESP",
        "message": <e.g. "start_teleoperation" or "stop">,
        "error": <any error string (or "None")>,
        ... plus any extra fields from kwargs ...
        }
    Then puts that response dict to the `teachbot_interface_commup` queue.
    """
    response = payload.copy()
    response["type"] = "RESP"
    if error not in ("None", ""):
        response["error"] = error
    elif response.get("error", "") == "":
        response["error"] = "None"

    # Add additional fields if present
    response.update(kwargs)

    # Log and publish
    logger_ti.info(f"Preparing to send response: {response}")
    teachbot_interface_commup.put(response)
    logger_ti.info(f"Sent response: {response}")


class DummyTeachbot(TeachbotInterface):
    """
    A dummy 'Teachbot' that simulates a single joint moving
    back and forth at ~100 Hz, and sends status messages
    in the same format as a real Teachbot would.
    """

    def __init__(self, teachbot_interface_commup, teachbot_interface_commdown,
                 shm_target_pos1, logger_ti, config):
        
        self.teachbot_interface_commup = teachbot_interface_commup
        self.teachbot_interface_commdown = teachbot_interface_commdown
        self.shm_target_pos1 = shm_target_pos1
        self.logger_ti = logger_ti
        self.config = config

        self.connected = False
        self.status_streaming = False
        self.joint_target_streaming = False

        # We’ll simulate 6 joints, although only joint 1 actually moves
        self.current_joints_deg = np.zeros(6, dtype=float)

        # Tweak to your refresh rate
        self.status_refresh_period = config["general"]["status_refresh_period"]  # e.g. 0.016
        self.joint_target_period = config["hardware"]["teachbot"]["joint_target_period"]  # e.g. 0.01

        # Movement direction/state
        self._direction = 1  # +1 or -1
        self.step_size = 30  # degrees per second
        self.random_degrees = [-1, 1]
        
        # State for the new movement pattern
        self._movement_sequence = [
            (0, -30),  # joint 1 to -30
            (2, -30),  # joint 3 to -30
            (0, 30),   # joint 1 to 30
            (2, 0),   # joint 3 to 30
        ]
        self._current_step = 0
        self._target_reached = False

        self._stop_flag = False
        self.status_thread = None
        self.joint_target_thread = None

        # Simulate connect
        self.connect()
        self.start_joint_target_streaming()

    ############################################################
    # Command functions
    ############################################################

    def stop(self):
        self.logger_ti.info("DummyTeachbot: Disconnect called")
        self.connected = False
        if self.joint_target_streaming:
            self.stop_joint_target_streaming()

    ############################################################
    # Joint target streaming
    ############################################################

    def connect(self):
        # Simulate a failing connection to demonstrate error handling
        possible_to_connect = True
        if not possible_to_connect:
            self.logger_ti.error("DummyTeachbot: Unable to connect")
            # Raise an exception so it can be caught by run_teachbot_interface(...)
            raise Exception("Unable to connect to DummyTeachbot")
        self.logger_ti.info("DummyTeachbot: Simulating connect()")
        self.connected = True
        return True


    def start_joint_target_streaming(self):
        self.joint_target_streaming = True
        self.joint_target_thread = threading.Thread(target=self._joint_target_thread_fn)
        self.joint_target_thread.start()
        return True

    def stop_joint_target_streaming(self):
        self.joint_target_streaming = False
        if self.joint_target_thread:
            self.joint_target_thread.join(timeout=1.0)
        self.joint_target_thread = None

    def _joint_target_thread_fn(self):
        """
        Moves joints in a specific sequence: joint1 to -30, joint3 to -30, 
        joint1 to 30, joint3 to 30, and repeats this pattern.
        Puts the 'current_joints_deg' array into shm_target_pos1 to simulate real robot data.
        """
        while self.joint_target_streaming:
            if self.connected:
                # Get current target from sequence
                target_joint_index, target_position = self._movement_sequence[self._current_step]
                current_position = self.current_joints_deg[target_joint_index]
                
                # Calculate movement towards target
                position_diff = target_position - current_position
                
                # Check if we've reached the target (within a small tolerance)
                if abs(position_diff) < 0.5:  # 0.5 degree tolerance
                    # Target reached, move to next step in sequence
                    self._current_step = (self._current_step + 1) % len(self._movement_sequence)
                    self._target_reached = True
                else:
                    # Move towards target
                    move_direction = 1 if position_diff > 0 else -1
                    movement_step = move_direction * self.step_size * self.joint_target_period
                    
                    # Don't overshoot the target
                    if abs(movement_step) > abs(position_diff):
                        movement_step = position_diff
                    
                    self.current_joints_deg[target_joint_index] += movement_step

                # Add random noise to all 6 joints before sending to shared memory
                noisy_joints = self.current_joints_deg.copy()
                for i in range(6):
                    random_factor = np.random.uniform(self.random_degrees[0], self.random_degrees[1])
                    noisy_joints[i] += random_factor

                # Push the joint values into the shared memory
                try:
                    if self.shm_target_pos1.full():
                        self.shm_target_pos1.get_nowait()
                    self.shm_target_pos1.put(noisy_joints.copy(), timeout=1)
                except:
                    pass  # ignoring any queue exceptions

            time.sleep(self.joint_target_period)


def run_teachbot_interface(teachbot_interface_commup, teachbot_interface_commdown, shm_target_pos1):
    # Load the config
    config = load_config()

    # Setup logging
    component_tag = "TEACHBOT_INTERFACE"
    logger_ti = setup_logging(component_tag)
    
    robot_brand = config["hardware"]["teachbot"]["brand"].lower().capitalize()
    check_queue_period = config["general"]["check_queue_period"]

    # Tell the controller that the interface is starting
    logger_ti.info("Starting %s Teachbot Interface", robot_brand)

    # Attempt to instantiate the correct class for the brand
    teachbot = None
    teachbot_class_name = f"{robot_brand}Teachbot"
    try:
        TeachbotClass = globals()[teachbot_class_name]
        teachbot = TeachbotClass(
            teachbot_interface_commup,
            teachbot_interface_commdown,
            shm_target_pos1,
            logger_ti,
            config
        )
        logger_ti.info("Successfully initialized %s class", teachbot_class_name)

        # Notify controller that initialization succeeded
        send_response(
            logger_ti,
            teachbot_interface_commup,
            {"interface": component_tag, "message": "initialization"},
            error="None"
        )
    except Exception as e:
        logger_ti.error("Error instantiating %s: %s", teachbot_class_name, e)
        error_msg = f"Error instantiating {teachbot_class_name}: {e}"
        send_response(
            logger_ti,
            teachbot_interface_commup,
            {"interface": component_tag, "message": "initialization"},
            error=error_msg
        )
        return

    # Main command loop
    while True:
        # Check for commands from the controller
        if not teachbot_interface_commdown.empty():
            full_message = teachbot_interface_commdown.get()
            logger_ti.info("Received message: %s", full_message)

            msg_type = full_message.get("type", "")
            msg_interface = full_message.get("interface", "")
            message = full_message.get("message", "")

            if msg_type == "CMD" and msg_interface == "TEACHBOT_INTERFACE":
                if message == "stop":
                    try:
                        if teachbot is None:
                            pass 
                        elif teachbot.connected:
                            teachbot.stop()
                            logger_ti.info("Disconnected from %s Teachbot", robot_brand)
                        else:
                            logger_ti.info("Already disconnected from %s Teachbot", robot_brand)
                        # Send success response
                        send_response(logger_ti, teachbot_interface_commup, full_message, error="None")
                    except Exception as ex_stop:
                        logger_ti.error("Error stopping Teachbot: %s", ex_stop)
                        send_response(logger_ti, teachbot_interface_commup, full_message, error=str(ex_stop))
                    # Exit the loop
                    break
                else:
                    logger_ti.warning("Unknown command: %s", full_message)
                    # Send a response indicating unknown command
                    send_response(logger_ti, teachbot_interface_commup, full_message, error="Unknown command")
            else:
                logger_ti.warning("Unknown message: %s", full_message)
                send_response(logger_ti, teachbot_interface_commup, full_message, error="Unknown message")

        # Small sleep to prevent CPU hogging
        time.sleep(check_queue_period)
