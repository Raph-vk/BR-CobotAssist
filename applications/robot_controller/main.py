#!/usr/bin/env python3
# Main entry point for the robot controller

"""
cd <project_root>/applications/robot_controller
python3 main.py
"""

from robot_controller import RobotController
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.utils import load_config, setup_logging


def main(config, logger_tc):
    global controller

    controller = RobotController(config=config, logger_tc=logger_tc)

    # Keep running until processes_running is false
    while controller.processes_running:
        time.sleep(0.5)
    while not controller.cleanup_done:
        time.sleep(0.5)

if __name__ == "__main__":
    # Load the config
    config = load_config()

    # Setup logging
    component_tag = "ROBOT_CONTROLLER"
    logger_tc = setup_logging(component_tag)

    main(config=config, logger_tc=logger_tc)
