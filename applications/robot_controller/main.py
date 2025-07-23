#!/usr/bin/env python3
# Main entry point for the robot controller

"""
cd <project_root>/applications/robot_controller
python3 main.py [--setup SETUP_ID]
"""

from robot_controller import RobotController
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.utils import load_config, setup_logging


def main(config, logger_tc, setup_id=None):
    global controller

    controller = RobotController(config=config, logger_tc=logger_tc, setup_id=setup_id)

    # Keep running until processes_running is false
    while controller.processes_running:
        time.sleep(0.5)
    while not controller.cleanup_done:
        time.sleep(0.5)

if __name__ == "__main__":
    # Simple command line argument handling
    setup_id = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--setup":
        setup_id = sys.argv[2]
    
    # Load the config
    config = load_config()

    # Setup logging with setup ID in component tag
    if setup_id:
        component_tag = f"ROBOT_CONTROLLER_SETUP_{setup_id}"
        print(f"Starting robot controller for setup {setup_id}")
    else:
        component_tag = "ROBOT_CONTROLLER"
        print("Starting robot controller (no specific setup)")
        
    logger_tc = setup_logging(component_tag)
    
    if setup_id:
        logger_tc.info(f"Starting robot controller for setup {setup_id}")
    else:
        logger_tc.info("Starting robot controller (no specific setup)")

    main(config=config, logger_tc=logger_tc, setup_id=setup_id)
