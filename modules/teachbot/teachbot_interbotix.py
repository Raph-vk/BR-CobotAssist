
import time
import copy
import numpy as np
import threading
import os
import sys
import signal
from threading import Thread

from .teachbot_0interface import TeachbotInterface

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.utils import setup_logging, load_config



#################################################################
# Helper Functions
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
# my_app/modules/teachbot_interbotix.py

def set_torque_for_joint(robot_name, joint_name, enable):
    # Import rospy and TorqueEnable only when needed (after ROS is sourced)
    import rospy
    from interbotix_xs_msgs.srv import TorqueEnable
    
    service_name = f'/{robot_name}/torque_enable'
    rospy.wait_for_service(service_name)
    try:
        torque_service = rospy.ServiceProxy(service_name, TorqueEnable)
        response = torque_service(cmd_type='single', name=joint_name, enable=enable)
        return response
    except rospy.ServiceException as e:
        print("Service call failed:", e)


def send_command(queue, command, last_command=False):
    queue.put(command)
    if last_command:
        queue.put(f"[CMD][TEACHBOT_INTERFACE]: ready")


def sent_ready(teachbot_interface_commup, logger_ti):
    """
    Helper function to send a ready message to the controller.
    """
    logger_ti.info("Sending ready message to controller.")
    teachbot_interface_commup.put("[CMD][TEACHBOT_INTERFACE]: ready")
    logger_ti.info("Ready message sent.")


#################################################################
# Teachbot Implementation
#################################################################

class InterbotixTeachbot(TeachbotInterface):
    """
    A 'Teachbot' specifically for Interbotix hardware 
    (records and executes paths via ROS).
    """

    def __init__(self, teachbot_interface_commup, teachbot_interface_commdown, shm_target_pos1, logger_ti, config):
        self.teachbot_interface_commup = teachbot_interface_commup
        self.teachbot_interface_commdown = teachbot_interface_commdown
        self.shm_target_pos1 = shm_target_pos1
        self.logger_ti = logger_ti
        self.config = config

        self.robot_model = config["hardware"]["teachbot"]["model"].lower()
        self.group_name = "arm"
        self.gripper_name = "gripper"
        self.robot_name = 'master_left'
        self.target_pos_period = config["hardware"]["robot"]["control_dt"]
        self.min_joint6 = config["hardware"]["teachbot"]["j6_min_angle"]
        self.max_joint6 = config["hardware"]["teachbot"]["j6_max_angle"]
        self.joint6_multiplier = config["hardware"]["teachbot"]["j6_multiplier"]
        self.joint4_locked = config["hardware"]["teachbot"]["j4_locked"]
        self.teachbot_gripper_min = config["hardware"]["teachbot"]["gripper_min"]
        self.teachbot_gripper_max = config["hardware"]["teachbot"]["gripper_max"]
        self.j6_factor = config["general"]["j6_factor"]

        self.bot = None
        self.publish_thread = None
        self.ros_process = None

        # Status variables
        self.connected = False
        self.joint_target_streaming = False

        self.connect()
        self.start_joint_target_streaming()

    ##########################################################
    # Commands
    ##########################################################

    def stop(self):
        if self.joint_target_streaming:
            self.stop_joint_target_streaming()
        if self.ros_process is not None:
            self.connected = False
            os.killpg(os.getpgid(self.ros_process.pid), signal.SIGINT)
            self.logger_ti.info("ROS process killed")
            self.ros_process = None
            self.bot = None


    ##########################################################
    # Helper functions
    ##########################################################

    def connect(self):
        """
        Connect to the InterbotixManipulatorXS robot.
        ROS environment should already be sourced by the launch script.
        """
        self.logger_ti.info("Connecting to Interbotix Teachbot...")
        
        # Import ROS packages (environment should already be sourced)
        try:
            import rospy
            import subprocess
            import os
            
            # Initialize ROS node
            rospy.init_node('interbotix_teachbot', disable_signals=True, disable_rosout=True)
            self.logger_ti.info("ROS node initialized")
            
            # Check if roscore is running
            try:
                result = subprocess.run(['rostopic', 'list'], capture_output=True, text=True, timeout=5)
                if result.returncode != 0:
                    self.logger_ti.error("roscore is not running. Please start roscore first.")
                    return False
            except subprocess.TimeoutExpired:
                self.logger_ti.error("rostopic command timed out. ROS may not be properly set up.")
                return False
            except FileNotFoundError:
                self.logger_ti.error("rostopic command not found. ROS may not be in PATH.")
                return False
            
            # Start roslaunch in the background
            launch_cmd = "roslaunch aloha 4arms_teleop.launch"
            self.ros_process = subprocess.Popen(
                launch_cmd,
                shell=True,
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give ROS time to start up
            self.logger_ti.info("Waiting for ROS launch to initialize...")
            time.sleep(5)
            
            self.logger_ti.info("ROS process started")
            
            # Connect to the robot in a separate thread
            def create_bot():
                try:
                    from interbotix_xs_modules.arm import InterbotixManipulatorXS
                    
                    self.bot = InterbotixManipulatorXS(
                        robot_model=self.robot_model,
                        group_name=self.group_name,
                        gripper_name=self.gripper_name,
                        robot_name=self.robot_name,
                        init_node=False
                    )
                    self.connect_exception = None
                except Exception as e:
                    self.connect_exception = e

            self.connect_exception = None
            t = threading.Thread(target=create_bot, daemon=True)
            t.start()

            self.logger_ti.info("Waiting for InterbotixManipulatorXS to connect...")
            
            # Wait up to 10 seconds for InterbotixManipulatorXS to finish initializing
            t.join(timeout=10.0)
            
            self.logger_ti.info("Finished waiting for InterbotixManipulatorXS to connect...")

            if t.is_alive():
                # Timed out
                self.logger_ti.error("Timed out waiting for %s %s to connect", self.robot_name, self.robot_model)
                if hasattr(self, 'ros_process') and self.ros_process:
                    os.killpg(os.getpgid(self.ros_process.pid), signal.SIGTERM)
                return False
            elif self.connect_exception is not None:
                # Some error occurred in creation
                self.logger_ti.error("Error connecting to %s %s: %s", self.robot_name, self.robot_model, self.connect_exception)
                if hasattr(self, 'ros_process') and self.ros_process:
                    os.killpg(os.getpgid(self.ros_process.pid), signal.SIGTERM)
                return False
            else:
                # Success
                self.logger_ti.info("Connected to %s %s", self.robot_name, self.robot_model)
                self.connected = True
                
                # Enable torque for joint if needed
                if self.joint4_locked:
                    set_torque_for_joint(self.robot_name, 'forearm_roll', True)
                    self.logger_ti.info("Torque enabled for forearm_roll")
                return True
                
        except ImportError as e:
            self.logger_ti.error("Error importing ROS modules: %s", e)
            self.logger_ti.error("Make sure ROS and Interbotix packages are installed and sourced")
            return False
        except Exception as e:
            self.logger_ti.error("Unexpected error during connection: %s", e)
            return False


    def start_joint_target_streaming(self):
        self.joint_target_streaming = True
        self.joint_target_streaming_thread = Thread(target=self.joint_target_updating)
        self.joint_target_streaming_thread.start()
        return True

    def stop_joint_target_streaming(self):
        self.joint_target_streaming = False
        self.joint_target_streaming_thread.join(0.5)
        self.joint_target_streaming_thread = None
        return True



    def joint_target_updating(self):
        while self.joint_target_streaming:
            if self.connected:
                try: 
                    # Get joints and gripper separately like in working code
                    joints = self.bot.dxl.joint_states.position[:6]  # Get first 6 joints
                    gripper = self.bot.dxl.joint_states.position[6]  # Get gripper (7th element)
                    
                    # Combine into 7-element array
                    joint_states = np.zeros(7)
                    joint_states[:6] = joints
                    joint_states[6] = gripper  # Use raw gripper value without normalization

                    
                    joint_states_translated, safe = self.action_translation_and_check(joint_states)

                    if safe:
                        if self.shm_target_pos1.full():
                            self.shm_target_pos1.get_nowait()
                        self.shm_target_pos1.put(joint_states_translated, timeout=0.1)
                    else:
                        self.logger_ti.warning("Joint target not safe")
                except Exception as e:
                    self.logger_ti.error("Error updating joint target: %s", e)
            time.sleep(self.target_pos_period)
    
    def action_translation_and_check(self, action):
       
        action = np.array(action)
        action = action * 180 / np.pi
        joint6_translated = action[5] * self.joint6_multiplier
        joint6_translated = max(self.min_joint6, min(joint6_translated, self.max_joint6))
        action[5] = joint6_translated
        if self.joint4_locked:
            action[3] = 0
        action[2] = -action[2]
        action[4] = -action[4]
        action[5] = -action[5]

        safe = True
        if action[0] < -175:
            if action[1] < 175:
                safe = False

        # Normalize the gripper value
        if action[-1] > self.teachbot_gripper_max:
            action[-1] = self.teachbot_gripper_max
        elif action[-1] < self.teachbot_gripper_min:
            action[-1] = self.teachbot_gripper_min
        action[-1] = (action[-1] - self.teachbot_gripper_min) / (self.teachbot_gripper_max - self.teachbot_gripper_min)        

        action[5] = action[5] * self.j6_factor

        return action, safe


#################################################################
# The main interface loop
#################################################################

def run_teachbot_interface(teachbot_interface_commup, teachbot_interface_commdown, shm_target_pos1, setup_id=None):
    # Load the config
    config = load_config()

    # Setup logging with setup-specific tag
    if setup_id:
        component_tag = f"{int(setup_id):02d}_TEACHBOT_INTERFACE"
    else:
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

            if msg_type == "CMD" and msg_interface == component_tag:
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
