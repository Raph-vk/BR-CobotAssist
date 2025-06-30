#!/usr/bin/env python3
# applications/tos_ui/main.py

import logging
import logging.config
import os
import subprocess
import webbrowser

import yaml

# Import your TOSUIApplication class
from applications.tos_ui.tos_ui import TOSUIApplication
from utils.utils import load_config, setup_logging



def focus_or_open_browser(url, logger):
    """
    Try to focus existing browser window with the URL if it exists,
    otherwise open a new browser window.
    """
    try:
        logger.info("Searching for 'TOS UI — Mozilla Firefox' window...")

        cmd = "wmctrl -l | grep -i 'TOS UI.*Firefox'"
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0 and result.stdout.strip():
            window_id = result.stdout.strip().split()[0]
            logger.info(f"Found TOS UI Firefox window: {window_id}")
            activate_cmd = f"wmctrl -ia {window_id}"
            subprocess.run(activate_cmd, shell=True)
            logger.info("Successfully brought TOS UI window to front")
            return
        else:
            logger.info("TOS UI Firefox window not found")

        if os.system('which xdotool >/dev/null 2>&1') == 0:
            logger.info("Trying xdotool to find TOS UI window...")
            xdotool_cmd = "xdotool search --onlyvisible --name 'TOS UI.*Firefox' windowactivate"
            xdotool_result = subprocess.run(xdotool_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if xdotool_result.returncode == 0:
                logger.info("Successfully activated TOS UI window with xdotool")
                return

        logger.info("No existing TOS UI browser window found, opening new one.")
        webbrowser.open_new(url)

    except Exception as e:
        logger.error(f"Error while trying to focus browser window: {e}")
        webbrowser.open_new(url)



def run(config, ui_logger):
    ui_logger.info("Starting TOS UI...")

    # Create and start the TOS UI Application, passing in our logger + config
    tos_ui = TOSUIApplication(ui_logger=ui_logger, config=config)

    # Build URL based on the config
    flask_host = config["application"]["flask_host"]
    flask_port = config["application"]["flask_port"]
    url = f"http://{flask_host}:{flask_port}"

    # Open the UI in the browser or focus existing window
    focus_or_open_browser(url, ui_logger)

    # Start the Flask app
    tos_ui.start(host=flask_host, port=flask_port)


if __name__ == "__main__":
    # Load the entire config from config.yaml
    config = load_config()

    # Setup logging from config
    ui_logger = setup_logging(component_tag="[UI]")

    # Run the main application logic
    run(config, ui_logger)
