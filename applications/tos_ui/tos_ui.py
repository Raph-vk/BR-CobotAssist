#!/usr/bin/env python3
# applications/tos_ui/tos_ui.py

"""
Example usage:
cd tos_app/applications/tos_ui
python3 main.py
"""

import os
import time
import json
import threading
import subprocess
from contextlib import contextmanager

from flask import Flask, render_template, request, redirect, url_for, jsonify
import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError

from utils.utils import robust_connect  # or keep robust_connect inline if you prefer


class TOSUIApplication:
    def __init__(self, ui_logger, config):
        self.app = Flask(__name__)
        self.ui_logger = ui_logger
        self.config = config

        # RabbitMQ + Application config
        self.rabbit_conf = config["rabbitmq"]
        self.app_conf = config["application"]

        # For consumer threads
        self.stop_flag = False
        self.recording_names = None
        self.recv_recording_names_status = False
        self.dataset_names = None
        self.recv_dataset_names_status = False
        self.model_names = None  # For model names if needed later
        self.recv_model_names_status = False

        # Prepare infrastructure once (exchange, queues, bindings)
        self.setup_rabbitmq_infrastructure()

        # Register Flask routes
        self.register_routes()

        # Start background consumer threads
        self._start_consumer_threads()

    ###################################################################
    # 1) Setup RabbitMQ (declare exchange, queues, bindings) once
    ###################################################################
    def setup_rabbitmq_infrastructure(self):
        self.ui_logger.info("Setting up RabbitMQ infrastructure...")
        with open_channel(self.rabbit_conf, self.ui_logger, client_name="ui_infra_setup") as channel:
            # Exchange
            channel.exchange_declare(
                exchange=self.rabbit_conf["exchange_name"],
                exchange_type=self.rabbit_conf["exchange_type"],
                durable=False
            )
            # Declare + bind the STATUS queue
            channel.queue_declare(
                queue=self.rabbit_conf["ui_queue_name"],
                durable=False,
                auto_delete=True
            )
            channel.queue_bind(
                exchange=self.rabbit_conf["exchange_name"],
                queue=self.rabbit_conf["ui_queue_name"],
                routing_key=self.rabbit_conf["status_binding_key"]
            )
            # Declare + bind the RESPONSE queue
            channel.queue_declare(
                queue=self.rabbit_conf["response_queue_name"],
                durable=False,
                auto_delete=True
            )
            response_binding = self.rabbit_conf["response_binding_key_prefix"] + "#"
            channel.queue_bind(
                exchange=self.rabbit_conf["exchange_name"],
                queue=self.rabbit_conf["response_queue_name"],
                routing_key=response_binding
            )
            self.ui_logger.info("RabbitMQ exchange/queues/bindings set up successfully.")

    ###################################################################
    # 2) Start Consumer Threads (Status & Response)
    ###################################################################
    def _start_consumer_threads(self):

        # Start the response consumer
        self.response_thread = threading.Thread(
            target=robust_consume,
            args=(
                self.rabbit_conf,
                self.ui_logger,
                self.rabbit_conf["response_queue_name"],
                self.rabbit_conf["response_binding_key_prefix"] + "#",
                self.on_response_message,
                lambda: self.stop_flag
            ),
            daemon=True
        )
        self.response_thread.start()
        self.ui_logger.info("Started RabbitMQ response consumer in background thread.")

    ###################################################################
    # 3) Flask Routing + Start
    ###################################################################
    def register_routes(self):
        @self.app.route("/")
        def index():
            return render_template("main_window.html")

        @self.app.route("/send_command", methods=["POST"])
        def handle_command():
            """
            Accepts a command_type and optional recording_name,
            sends them to RabbitMQ, and returns JSON (no page reload).
            """
            message = request.form.get("message", "")
            recording_name = request.form.get("recording_name", "")
            dataset_name = request.form.get("dataset_name", "")
            model_name = request.form.get("model_name", "")
            recording_speed = float(request.form.get("recording_speed", "")) if request.form.get("recording_speed") else 0.0
            playback_speed = float(request.form.get("playback_speed", "")) if request.form.get("playback_speed") else 0.0

            self.ui_logger.info(f"Received form data - message: '{message}', dataset_name: '{dataset_name}', recording_name: '{recording_name}'")

            if not message:
                return jsonify({"status": "error", "message": "No message specified"}), 400

            # Build a dict for the message, with the message and optional recording_name
            msg = {"type": "CMD", "message": message}
            if message == "play_recording":
                msg["recording_name"] = recording_name
                msg["playback_speed"] = playback_speed
            elif message == "delete_recording":
                msg["recording_name"] = recording_name

            elif message == "start_teleoperation_record":
                # generate recording name based on current date/time
                recording_name = time.strftime("%Y%m%d_%H%M%S") + ".json"
                msg["recording_name"] = recording_name
                msg["recording_speed"] = recording_speed




            elif message == "record_episodes":
                # If no dataset name is selected, create a new one based on timestamp
                if not dataset_name:
                    dataset_name = time.strftime("%Y%m%d_%H%M%S")
                msg["dataset_name"] = dataset_name
                msg["recording_speed"] = recording_speed

            elif message == "record_episode":
                # If no dataset name is selected, create a new one based on timestamp
                if not dataset_name:
                    dataset_name = time.strftime("%Y%m%d_%H%M%S")
                msg["dataset_name"] = dataset_name
                msg["recording_speed"] = recording_speed

            elif message == "record_mistake":
                msg["dataset_name"] = dataset_name

            elif message == "run_policy":
                msg["dataset_name"] = dataset_name
                msg["model_name"] = model_name

            elif message == "train_policy":
                msg["dataset_name"] = dataset_name

            elif message == "start_teleoperation":
                msg["recording_speed"] = recording_speed

            # Send the command via RabbitMQ
            self.send_command(msg)
            self.ui_logger.info("Command sent: %s", msg)

            # Return JSON so we don't reload the page
            return jsonify({
                "status": "ok",
                "message_sent": message,
                "recording_name": recording_name if recording_name else None
            })

        @self.app.route("/request_recordings", methods=["POST"])
        def request_recordings():
            """
            1) Publish the command 'report_recording_names'
            2) Wait up to 2 seconds for on_response_message to fill self.recording_names
            3) Return the filenames as JSON (no redirect).
            """
            self.recv_recording_names_status = False

            # Note we include type="CMD"
            msg = {"type": "CMD", "message": "report_recording_names"}
            self.send_command(msg)

            start_time = time.time()
            while time.time() - start_time < 2.0:
                if self.recv_recording_names_status:
                    break
                time.sleep(0.01)

            # Return whatever we have (could be empty if no names arrived in time)
            return jsonify(self.recording_names or [])

        @self.app.route("/request_datasets", methods=["POST"])
        def request_datasets():
            """
            1) Publish the command 'report_dataset_names'
            2) Wait up to 2 seconds for on_response_message to fill self.dataset_names
            3) Return the dataset names as JSON (no redirect).
            """
            self.recv_dataset_names_status = False

            # Note we include type="CMD"
            msg = {"type": "CMD", "message": "report_dataset_names"}
            self.send_command(msg)

            start_time = time.time()
            while time.time() - start_time < 2.0:
                if self.recv_dataset_names_status:
                    break
                time.sleep(0.01)

            # Return whatever we have (could be empty if no names arrived in time)
            return jsonify(self.dataset_names or [])

        @self.app.route("/request_models", methods=["POST"])
        def request_models():
            """
            1) Publish the command 'report_model_names'
            2) Wait up to 2 seconds for on_response_message to fill self.model_names
            3) Return the model names as JSON (no redirect).
            """
            self.recv_model_names_status = False

            dataset_name = request.form.get("dataset_name", "")
            self.ui_logger.info(f"Retrieved dataset_name for models: '{dataset_name}'")

            # Note we include type="CMD"
            msg = {"type": "CMD", "message": "report_model_names", "dataset_name": dataset_name}
            self.send_command(msg)

            start_time = time.time()
            while time.time() - start_time < 2.0:
                if self.recv_model_names_status:
                    break
                time.sleep(0.01)

            # Return whatever we have (could be empty if no names arrived in time)
            return jsonify(self.model_names or [])


    def start(self, host, port):
        """Start the Flask application."""
        flask_debug = self.app_conf["flask_debug"]
        flask_use_reloader = self.app_conf["flask_use_reloader"]
        self.app.run(host=host, port=port, debug=flask_debug, use_reloader=flask_use_reloader)

    ###################################################################
    # 4) Publisher Method (Send Commands)
    ###################################################################
    def send_command(self, cmd):
        """
        Publish a JSON message to RabbitMQ with a routing key
        """
        ensure_controller_running(
            self.ui_logger,
            open_terminal=self.app_conf["open_terminal_default"],
            controller_path=self.app_conf["controller_path"],
            rabbit_conf=self.rabbit_conf,
            config=self.config
        )

        # Derive the routing key suffix from the "message" field if present
        if isinstance(cmd, dict) and "message" in cmd:
            routing_key_suffix = cmd["message"]
        else:
            routing_key_suffix = "unknown"

        cmd_json = json.dumps(cmd)  # the actual message body
        routing_key = f"{self.rabbit_conf['command_binding_key_prefix']}{routing_key_suffix}"

        # Publish
        publish_message(
            rabbit_conf=self.rabbit_conf,
            ui_logger=self.ui_logger,
            routing_key=routing_key,
            message=cmd_json,
            client_name=self.rabbit_conf["client_name_ui_sender"]
        )
        self.ui_logger.info("Sent command: %s => routing_key='%s'", cmd_json, routing_key)

    ###################################################################
    # 5) Consumer Callbacks
    ###################################################################
    def on_response_message(self, ch, method, properties, body):
        """
        Handle messages from the response queue.
        The new response format is:
        {
        "type": "RESP",
        "message": "<command_name>",
        ...other fields like "recording_name", "recording_speed", "files", "error"...
        }
        """
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.ui_logger.warning("Invalid JSON in response message.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        # Log the entire response
        self.ui_logger.info("Received response: %s", data)

        # If there is an error field, check if it's truly an error
        if "error" in data:
            err_val = data["error"]
            # We only treat it as an error if it's not None, empty string, or the literal string "None"
            if err_val and err_val not in ("None", ""):
                self.ui_logger.error("Response error: %s", err_val)

        msg_type = data.get("type", "")
        message_cmd = data.get("message", "")

        # Check if it's indeed a "RESP"
        if msg_type == "RESP":
            # For "report_recording_names" if the controller sends that
            if message_cmd == "report_recording_names":
                # The new format includes "files": [ {...}, {...} ]
                files_list = data.get("files", [])
                if isinstance(files_list, list):
                    self.recording_names = files_list
                    self.recv_recording_names_status = True
                    self.ui_logger.info("Recording files: %s", self.recording_names)
                else:
                    self.ui_logger.warning("'report_recording_names' response has no 'files' list.")
            # For "report_dataset_names" if the controller sends that
            elif message_cmd == "report_dataset_names":
                # The new format includes "files": [ {...}, {...} ]
                files_list = data.get("files", [])
                if isinstance(files_list, list):
                    self.dataset_names = files_list
                    self.recv_dataset_names_status = True
                    self.ui_logger.info("Dataset files: %s", self.dataset_names)
                else:
                    self.ui_logger.warning("'report_dataset_names' response has no 'files' list.")
            # For "report_model_names" if the controller sends that
            elif message_cmd == "report_model_names":
                # The new format includes "files": [ {...}, {...} ]
                files_list = data.get("files", [])
                if isinstance(files_list, list):
                    self.model_names = files_list
                    self.recv_model_names_status = True
                    self.ui_logger.info("Model files: %s", self.model_names)
                else:
                    self.ui_logger.warning("'report_model_names' response has no 'files' list.")
            else:
                # For all other commands, just log them
                self.ui_logger.info("Response for '%s' command: %s", message_cmd, data)
        else:
            self.ui_logger.warning("Unknown response type: %s", msg_type)

        # Acknowledge that we processed the message
        ch.basic_ack(delivery_tag=method.delivery_tag)


    ###################################################################
    # 6) Stopping/Shutdown
    ###################################################################
    def stop_consumer(self):
        """
        Signal the consumer threads to stop and join them.
        """
        self.stop_flag = True
        if hasattr(self, 'status_thread'):
            self.status_thread.join(timeout=2.0)
        if hasattr(self, 'response_thread'):
            self.response_thread.join(timeout=2.0)
        self.ui_logger.info("All consumer threads stopped.")


###################################################################
# Helper Functions
###################################################################

@contextmanager
def open_channel(rabbit_conf, ui_logger, client_name=None):
    """
    Context manager that yields a Pika channel, then cleans it up.
    """
    connection = robust_connect(rabbit_conf, ui_logger, client_name=client_name)
    channel = connection.channel()
    try:
        yield channel
    finally:
        channel.close()
        connection.close()

def publish_message(rabbit_conf, ui_logger, routing_key, message, client_name=None):
    """
    Ephemeral publish: opens a channel, publishes, then closes.
    """
    with open_channel(rabbit_conf, ui_logger, client_name=client_name) as channel:
        channel.basic_publish(
            exchange=rabbit_conf["exchange_name"],
            routing_key=routing_key,
            body=message.encode("utf-8") if isinstance(message, str) else message
        )

def robust_consume(rabbit_conf, ui_logger, queue_name, routing_key, on_message_callback, stop_flag_func):
    """
    Repeatedly connect -> consume from queue_name -> on_message_callback
    If disconnected, it tries again after a short sleep.
    
    :param rabbit_conf: your rabbitmq config dict
    :param ui_logger: logger
    :param queue_name: which queue to consume from
    :param routing_key: the routing key used (for informational logs only)
    :param on_message_callback: function(ch, method, properties, body)
    :param stop_flag_func: a callable (lambda) returning True if we should stop.
    """
    ui_logger.info("Starting robust_consume on queue='%s' with routing_key='%s'.", queue_name, routing_key)
    while not stop_flag_func():
        try:
            # 1) Connect + channel
            connection = robust_connect(rabbit_conf, ui_logger, client_name=f"ui_consumer_{queue_name}")
            channel = connection.channel()

            # 2) Start consuming
            channel.basic_consume(
                queue=queue_name,
                on_message_callback=on_message_callback,
                auto_ack=False
            )
            ui_logger.info("Begin consuming on queue='%s' with routing_key='%s'.", queue_name, routing_key)
            channel.start_consuming()

        except (AMQPConnectionError, AMQPChannelError) as e:
            ui_logger.warning("Lost connection to RabbitMQ: {%s}. Reconnecting in 1s...", e)
            time.sleep(1)
        except Exception as e:
            ui_logger.error("Unexpected error in robust_consume: {%s}. Retrying in 1s...", e)
            time.sleep(1)
        finally:
            # If channel/connection are open, close them
            try:
                channel.close()
            except:
                pass
            try:
                connection.close()
            except:
                pass

        # If we're here, we either lost connection or were forcibly stopped
        if stop_flag_func():
            ui_logger.info("Stop flag detected. Exiting consume loop on queue='%s'.", queue_name)
            break

def ensure_controller_running(ui_logger, open_terminal, controller_path, rabbit_conf, config):
    """
    Checks if the controller is running; if not, attempts to launch it.
    If 'open_terminal' is True, opens a new terminal to run it.
    Otherwise, runs the controller in the background.
    Then waits for the controller's command queue to appear.
    """
    try:
        # Check if "robot_controller/main.py" is in the process list.
        subprocess.check_output(["pgrep", "-f", "robot_controller/main.py"])
        ui_logger.info("Controller seems to be running already.")
    except subprocess.CalledProcessError:
        ui_logger.info("No controller found. Launching main.py...")

        # Check if we need ROS environment based on hardware configuration
        needs_ros = False
        try:
            robot_brand = config.get("hardware", {}).get("robot", {}).get("brand", "").lower()
            teachbot_brand = config.get("hardware", {}).get("teachbot", {}).get("brand", "").lower()
            if robot_brand == "interbotix" or teachbot_brand == "interbotix":
                needs_ros = True
                ui_logger.info("Interbotix hardware detected. Launching with ROS environment.")
            else:
                ui_logger.info("No Interbotix hardware detected. Launching without ROS environment.")
        except Exception as e:
            ui_logger.warning("Could not check hardware configuration: %s. Launching without ROS.", e)

        if needs_ros:
            if open_terminal:
                subprocess.Popen([
                    "gnome-terminal", "--window", "--", "bash", "-c",
                    f"source /opt/ros/noetic/setup.bash && source $HOME/TOS/devel/setup.bash && python3 {controller_path}; exec bash"
                ])
                ui_logger.info("Launched controller in a new terminal with ROS environment.")
            else:
                # For background launch, use bash to source environments first
                subprocess.Popen([
                    "bash", "-c", 
                    f"source /opt/ros/noetic/setup.bash && source $HOME/TOS/devel/setup.bash && python3 {controller_path}"
                ])
                ui_logger.info("Launched controller in the background with ROS environment.")
        else:
            if open_terminal:
                subprocess.Popen([
                    "gnome-terminal", "--window", "--", "bash", "-c",
                    f"python3 {controller_path}; exec bash"
                ])
                ui_logger.info("Launched controller in a new terminal without ROS environment.")
            else:
                subprocess.Popen([
                    "python3", controller_path
                ])
                ui_logger.info("Launched controller in the background without ROS environment.")

    # Now wait until the controller's command consumer is active
    while True:
        try:
            # Just check existence of the queue (passive declare)
            with open_channel(rabbit_conf, ui_logger) as ch:
                ch.queue_declare(
                    queue=rabbit_conf["command_queue_name"],
                    passive=True, 
                    durable=False
                )
            ui_logger.info("Confirmed controller's command queue is active. Proceeding.")
            break
        except Exception as e:
            ui_logger.info("Waiting for controller's command consumer to come up... %s", e)
            time.sleep(1)
