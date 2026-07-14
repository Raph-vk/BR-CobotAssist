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

        # A single event is enough for the Teachbot UI: only one status request
        # is active at a time and the normal RabbitMQ consumer fills the result.
        self.connection_status = None
        self.connection_status_event = threading.Event()
        
        # Error message queue for frontend notifications
        self.error_messages = []
        self.error_lock = threading.Lock()

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
        self.ui_logger.info("Setting up RabbitMQ infrastructure for multi-setup...")
        with open_channel(self.rabbit_conf, self.ui_logger, client_name="ui_infra_setup") as channel:
            # Exchange
            channel.exchange_declare(
                exchange=self.rabbit_conf["exchange_name"],
                exchange_type=self.rabbit_conf["exchange_type"],
                durable=False
            )
            
            # Declare + bind the UI STATUS queue to receive from all setups
            ui_queue_name = self.rabbit_conf["ui_status_queue_name"]
            channel.queue_declare(
                queue=ui_queue_name,
                durable=False,
                auto_delete=True
            )
            
            # Bind UI queue to receive status and responses from all setups
            binding_patterns = self.rabbit_conf["ui_status_binding_patterns"]
            for pattern in binding_patterns:
                channel.queue_bind(
                    exchange=self.rabbit_conf["exchange_name"],
                    queue=ui_queue_name,
                    routing_key=pattern
                )
                self.ui_logger.info(f"UI queue bound to pattern: {pattern}")
            
            self.ui_logger.info("Multi-setup RabbitMQ infrastructure ready")

    ###################################################################
    # 2) Start Consumer Threads (Status & Response)
    ###################################################################
    def _start_consumer_threads(self):

        # Start the status/response consumer for all setups
        self.status_thread = threading.Thread(
            target=robust_consume,
            args=(
                self.rabbit_conf,
                self.ui_logger,
                self.rabbit_conf["ui_status_queue_name"],
                "robot_controller.*.#",  # Receive all messages from all setups
                self.on_status_message,
                lambda: self.stop_flag
            ),
            daemon=True
        )
        self.status_thread.start()
        self.ui_logger.info("Started RabbitMQ status/response consumer for all setups")

    ###################################################################
    # 3) Flask Routing + Start
    ###################################################################
    def register_routes(self):
        @self.app.route('/')
        def index():
            """Main page with dynamic template selection based on application type and number of robots"""
            
            try:
                # Get setup info from config
                setup_info = self.get_setup_info()
                
                # Check application type from config
                application_type = self.config.get("general", {}).get("application", "AI_robot_controller")
                
                # Get total number of setups defined in config
                total_setups_count = self.get_total_setups_count()
                
                # Get default recording speed from config
                default_recording_speed = self.config.get("general", {}).get("default_recording_speed", 0.5)
                
                # Get UI scale factor from config
                ui_scale_factor = self.config.get("application", {}).get("ui_scale_factor", 1.0)
                
                self.ui_logger.debug(f"Application type: {application_type}")
                self.ui_logger.debug(f"setup_info is: {setup_info}")
                self.ui_logger.debug(f"Total setups defined in config: {total_setups_count}")
                self.ui_logger.debug(f"Default recording speed: {default_recording_speed}")
                self.ui_logger.debug(f"UI scale factor: {ui_scale_factor}")
                
                # Choose template based on application type
                if application_type == "Teachbot_controller":
                    # Teachbot mode - validate single setup requirement
                    if len(setup_info) > 1:
                        error_msg = f"Error: Teachbot_controller mode requires exactly one active setup, but {len(setup_info)} setups are configured: {[s['name'] for s in setup_info]}"
                        self.ui_logger.error(error_msg)
                        # Return error page or fallback
                        return f"<h1>Configuration Error</h1><p>{error_msg}</p><p>Please configure exactly one setup in active_setups for Teachbot mode.</p>"
                    
                    # Check for customer-specific template
                    customer = self.config.get("general", {}).get("customer", "")
                    self.ui_logger.info(f"Customer from config: {customer}")
                    self.ui_logger.info(f"Flask template folder: {self.app.template_folder}")
                    if customer and not customer == "TOS":
                        template_name = f'teachbot_{customer}.html'
                    else:
                        template_name = 'teachbot.html'
                        self.ui_logger.info("No specific customer specified in config. Using default TOS teachbot.html template.")
                    
                    self.ui_logger.info(f"Using Teachbot mode with setup: {setup_info[0]['name']}")
                    
                    # Pass setup info and total setup count to template
                    return render_template(template_name, setups=setup_info, total_setups=total_setups_count, default_recording_speed=default_recording_speed, ui_scale_factor=ui_scale_factor)
                    
                elif application_type == "AI_robot_controller":
                    # AI robot controller mode - choose based on number of setups
                    if len(setup_info) <= 1:
                        # Single robot - use simplified template
                        template_name = '1_robot.html'
                    else:
                        # Multiple robots - use multi-robot template with setup selection
                        template_name = 'n_robots.html'
                    
                    self.ui_logger.info(f"Using AI robot controller mode with {len(setup_info)} setups")
                else:
                    # Unknown application type - default to AI mode
                    self.ui_logger.warning(f"Unknown application type '{application_type}', defaulting to AI_robot_controller mode")
                    if len(setup_info) <= 1:
                        template_name = '1_robot.html'
                    else:
                        template_name = 'n_robots.html'
                
                self.ui_logger.debug(f"Using template: {template_name} for {len(setup_info)} setups")
                
                # Pass setup info and total setup count to the template
                return render_template(template_name, setups=setup_info, total_setups=total_setups_count, default_recording_speed=default_recording_speed, ui_scale_factor=ui_scale_factor)
            except Exception as e:
                self.ui_logger.error(f"Error in index route: {e}")
                import traceback
                self.ui_logger.error(traceback.format_exc())
                # Provide a fallback with a simple info object
                fallback_setup = [{'name': 'Default', 'setup_id': '1', 'id': '1', 'display_name': 'Setup 1'}]
                return render_template('1_robot.html', setups=fallback_setup, total_setups=1, default_recording_speed=0.5, ui_scale_factor=1.0)
             
        
        @self.app.route("/send_command", methods=["POST"])
        def handle_command():
            """
            Accepts a command_type and optional recording_name,
            sends them to RabbitMQ, and returns JSON (no page reload).
            Now supports robot_setups parameter to target multiple setups.
            """
            message = request.form.get("message", "")
            recording_name = request.form.get("recording_name", "")
            dataset_name = request.form.get("dataset_name", "")
            model_name = request.form.get("model_name", "")
            old_recording_name = request.form.get("old_recording_name", "")
            new_recording_name = request.form.get("new_recording_name", "")
            old_dataset_name = request.form.get("old_dataset_name", "")
            new_dataset_name = request.form.get("new_dataset_name", "")
            old_model_name = request.form.get("old_model_name", "")
            new_model_name = request.form.get("new_model_name", "")
            recording_speed = float(request.form.get("recording_speed", "")) if request.form.get("recording_speed") else 0.0
            playback_speed = float(request.form.get("playback_speed", "")) if request.form.get("playback_speed") else 0.0
            extra_function1 = request.form.get("extra_function1", "false").lower() == "true"  # Convert string to boolean

            # Fixed sanding pressure is only applied by the robot interface for
            # start_teleoperation_record and play_recording. The UI sends a normalized
            # VPPE balance setpoint: neutral=0.5, with force in either direction.
            def parse_float_form(name, default):
                try:
                    return float(request.form.get(name, default))
                except (TypeError, ValueError):
                    self.ui_logger.warning("Invalid float for %s; using default %s", name, default)
                    return float(default)

            fixed_pressure_enabled = request.form.get("fixed_pressure_enabled", "false").lower() == "true"
            fixed_pressure_value = parse_float_form("fixed_pressure_value", 0.5)
            fixed_pressure_ui_percent = parse_float_form("fixed_pressure_ui_percent", 0.0)
            fixed_pressure_neutral = parse_float_form("fixed_pressure_neutral", 0.5)
            fixed_pressure_trigger_threshold = parse_float_form("fixed_pressure_trigger_threshold", 0.2)
            
            # Handle multiple robot setups (comma-separated) or single setup_id for backward compatibility
            robot_setups_str = request.form.get("robot_setups", "")
            setup_id = request.form.get("setup_id", "1")  # Fallback for backward compatibility
            
            if robot_setups_str:
                # Parse comma-separated robot setups
                target_setups = [setup.strip() for setup in robot_setups_str.split(",") if setup.strip()]
            else:
                # Fall back to single setup_id for backward compatibility
                target_setups = [setup_id]

            self.ui_logger.info(f"Received form data - message: '{message}', target_setups: {target_setups}, dataset_name: '{dataset_name}', recording_name: '{recording_name}', old_recording_name: '{old_recording_name}', new_recording_name: '{new_recording_name}', old_dataset_name: '{old_dataset_name}', new_dataset_name: '{new_dataset_name}', old_model_name: '{old_model_name}', new_model_name: '{new_model_name}', model_name: '{model_name}', recording_speed: {recording_speed}, playback_speed: {playback_speed}, extra_function1: {extra_function1}, fixed_pressure_enabled: {fixed_pressure_enabled}, fixed_pressure_value: {fixed_pressure_value}")

            if not message:
                return jsonify({"status": "error", "message": "No message specified"}), 400

            if not target_setups:
                return jsonify({"status": "error", "message": "No robot setups selected"}), 400

            # Build a dict for the message, with the message and optional recording_name
            msg = {"type": "CMD", "message": message}

            # Keep all fixed-pressure fields together so recording and playback receive
            # the same process parameters. Other commands intentionally ignore them.
            fixed_pressure_payload = {
                "fixed_pressure_enabled": fixed_pressure_enabled,
                "fixed_pressure_value": fixed_pressure_value,
                "fixed_pressure_ui_percent": fixed_pressure_ui_percent,
                "fixed_pressure_neutral": fixed_pressure_neutral,
                "fixed_pressure_trigger_threshold": fixed_pressure_trigger_threshold,
            }

            if message == "play_recording":
                msg["recording_name"] = recording_name
                msg["playback_speed"] = playback_speed
                msg["extra_function1"] = extra_function1
                msg.update(fixed_pressure_payload)
            elif message == "delete_recording":
                msg["recording_name"] = recording_name
            elif message == "rename_recording":
                msg["old_recording_name"] = old_recording_name
                msg["new_recording_name"] = new_recording_name
            elif message == "rename_dataset":
                msg["old_dataset_name"] = old_dataset_name
                msg["new_dataset_name"] = new_dataset_name
            elif message == "rename_model":
                msg["old_model_name"] = old_model_name
                msg["new_model_name"] = new_model_name
                msg["dataset_name"] = dataset_name
            elif message == "delete_dataset":
                msg["dataset_name"] = dataset_name
            elif message == "delete_model":
                msg["model_name"] = model_name
                msg["dataset_name"] = dataset_name

            elif message == "start_teleoperation_record":
                # Use the recording name from the frontend if provided, otherwise generate one
                if not recording_name:
                    recording_name = time.strftime("%Y%m%d_%H%M%S") + ".json"
                else:
                    # Ensure the recording name has .json extension
                    if not recording_name.endswith('.json'):
                        recording_name = recording_name + '.json'
                msg["recording_name"] = recording_name
                msg["recording_speed"] = recording_speed
                msg["extra_function1"] = extra_function1
                msg.update(fixed_pressure_payload)

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
                msg["model_name"] = model_name

            elif message == "start_teleoperation":
                msg["recording_speed"] = recording_speed
                msg["extra_function1"] = extra_function1

            # Send the command via RabbitMQ to all selected setups
            sent_to_setups = []
            for setup in target_setups:
                try:
                    self.send_command(msg, setup)
                    sent_to_setups.append(setup)
                    self.ui_logger.info("Command sent to setup %s: %s", setup, msg)
                except Exception as e:
                    self.ui_logger.error("Failed to send command to setup %s: %s", setup, e)

            # Return JSON so we don't reload the page
            return jsonify({
                "status": "ok",
                "message_sent": message,
                "sent_to_setups": sent_to_setups,
                "recording_name": recording_name if recording_name else None
            })

        @self.app.route("/request_recordings", methods=["POST"])
        def request_recordings():
            """
            1) Publish the command 'report_recording_names' to all available setups
            2) Wait up to 2 seconds for responses from all setups
            3) Return the aggregated recording names as JSON
            """
            self.recv_recording_names_status = False
            self.recording_names = []  # Reset recording names
            
            available_setup_ids = self.get_available_setup_ids()
            self.ui_logger.info(f"Requesting recordings from setups: {available_setup_ids}")

            # Send command to all available setups
            msg = {"type": "CMD", "message": "report_recording_names"}
            for setup_id in available_setup_ids:
                try:
                    self.send_command(msg, setup_id=setup_id)
                    self.ui_logger.debug(f"Sent report_recording_names to setup {setup_id}")
                except Exception as e:
                    self.ui_logger.warning(f"Failed to send report_recording_names to setup {setup_id}: {e}")

            start_time = time.time()
            while time.time() - start_time < 3.0:  # Increased timeout for multiple setups
                if self.recv_recording_names_status:
                    break
                time.sleep(0.01)

            # Return whatever we have (could be empty if no responses arrived in time)
            result = self.recording_names or []
            self.ui_logger.info(f"Returning {len(result)} recording names from all setups")
            return jsonify(result)

        @self.app.route("/request_connection_status", methods=["POST"])
        def request_connection_status():
            """Request robot status without changing the existing command API."""
            self.connection_status = None
            self.connection_status_event.clear()

            setup_ids = self.get_available_setup_ids()
            if not setup_ids:
                return jsonify({"connected": False, "operation": "Unavailable"})

            try:
                self.send_command(
                    {"type": "CMD", "message": "report_connection_status"},
                    setup_id=setup_ids[0],
                )
                self.connection_status_event.wait(timeout=3.0)
            except Exception as exc:
                self.ui_logger.warning("Connection status request failed: %s", exc)

            return jsonify(self.connection_status or {
                "connected": False,
                "plc_connected": False,
                "operation": "Unavailable",
            })

        @self.app.route("/request_datasets", methods=["POST"])
        def request_datasets():
            """
            1) Publish the command 'report_dataset_names' to all available setups
            2) Wait up to 2 seconds for responses from all setups
            3) Return the aggregated dataset names as JSON
            """
            self.recv_dataset_names_status = False
            self.dataset_names = []  # Reset dataset names
            
            available_setup_ids = self.get_available_setup_ids()
            self.ui_logger.info(f"Requesting datasets from setups: {available_setup_ids}")
            
            # Send command to all available setups
            msg = {"type": "CMD", "message": "report_dataset_names"}
            for setup_id in available_setup_ids:
                try:
                    self.send_command(msg, setup_id=setup_id)
                    self.ui_logger.debug(f"Sent report_dataset_names to setup {setup_id}")
                except Exception as e:
                    self.ui_logger.warning(f"Failed to send report_dataset_names to setup {setup_id}: {e}")

            start_time = time.time()
            while time.time() - start_time < 3.0:  # Increased timeout for multiple setups
                if self.recv_dataset_names_status:
                    break
                time.sleep(0.01)

            # Return whatever we have (could be empty if no responses arrived in time)
            result = self.dataset_names or []
            self.ui_logger.info(f"Returning {len(result)} dataset names from all setups")
            return jsonify(result)

        @self.app.route("/request_models", methods=["POST"])
        def request_models():
            """
            1) Publish the command 'report_model_names' to all available setups
            2) Wait up to 2 seconds for responses from all setups
            3) Return the aggregated model names as JSON
            """
            self.recv_model_names_status = False
            self.model_names = []  # Reset model names

            dataset_name = request.form.get("dataset_name", "")
            self.ui_logger.info(f"Retrieved dataset_name for models: '{dataset_name}'")
            
            available_setup_ids = self.get_available_setup_ids()
            self.ui_logger.info(f"Requesting models from setups: {available_setup_ids}")

            # Send command to all available setups
            msg = {"type": "CMD", "message": "report_model_names", "dataset_name": dataset_name}
            for setup_id in available_setup_ids:
                try:
                    self.send_command(msg, setup_id=setup_id)
                    self.ui_logger.debug(f"Sent report_model_names to setup {setup_id}")
                except Exception as e:
                    self.ui_logger.warning(f"Failed to send report_model_names to setup {setup_id}: {e}")

            start_time = time.time()
            while time.time() - start_time < 3.0:  # Increased timeout for multiple setups
                if self.recv_model_names_status:
                    break
                time.sleep(0.01)

            # Return whatever we have (could be empty if no responses arrived in time)
            result = self.model_names or []
            self.ui_logger.info(f"Returning {len(result)} model names from all setups")
            return jsonify(result)

        @self.app.route("/get_error_messages", methods=["GET"])
        def get_error_messages():
            """Get pending error messages for frontend display."""
            try:
                messages = self.get_error_messages()
                return jsonify({"success": True, "errors": messages})
            except Exception as e:
                self.ui_logger.error(f"Error retrieving error messages: {e}")
                return jsonify({"success": False, "errors": []})


    def start(self, host, port):
        """Start the Flask application."""
        flask_debug = self.app_conf["flask_debug"]
        flask_use_reloader = self.app_conf["flask_use_reloader"]
        self.app.run(host=host, port=port, debug=flask_debug, use_reloader=flask_use_reloader)

    ###################################################################
    # 4) Publisher Method (Send Commands)
    ###################################################################
    def send_command(self, cmd, setup_id=None):
        """
        Publish a JSON message to RabbitMQ with a routing key
        setup_id: specific setup to send to, or None for backward compatibility
        """
        ensure_controller_running(
            self.ui_logger,
            open_terminal=self.app_conf["open_terminal_default"],
            controller_path=self.app_conf["controller_path"],
            rabbit_conf=self.rabbit_conf,
            config=self.config,
            setup_id=setup_id
        )

        # Derive the routing key suffix from the "message" field if present
        if isinstance(cmd, dict) and "message" in cmd:
            routing_key_suffix = cmd["message"]
        else:
            routing_key_suffix = "unknown"

        cmd_json = json.dumps(cmd)  # the actual message body
        
        # Use new multi-setup routing pattern: robot_controller.{setup_id}.command.{action}
        if setup_id:
            routing_key = f"robot_controller.{setup_id}.command.{routing_key_suffix}"
        else:
            # Fallback to old pattern for backward compatibility
            routing_key = f"robot_controller.command.{routing_key_suffix}"

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
    def on_status_message(self, ch, method, properties, body):
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
            if (err_val and err_val not in ("None", "")
                    and data.get("message") != "report_connection_status"):
                self.ui_logger.error("Response error: %s", err_val)
                # Add error to the queue for frontend notification
                self._add_error_message(err_val)

        msg_type = data.get("type", "")
        message_cmd = data.get("message", "")

        # Check if it's indeed a "RESP"
        if msg_type == "RESP":
            if message_cmd == "report_connection_status":
                # Keep only the small, stable contract consumed by the badge.
                self.connection_status = {
                    "connected": bool(data.get("connected", False)),
                    "feedback_recent": data.get("feedback_recent"),
                    "plc_connected": bool(data.get("plc_connected", False)),
                    "operation": data.get("operation", "Unavailable"),
                }
                self.connection_status_event.set()
            # For "report_recording_names" if the controller sends that
            elif message_cmd == "report_recording_names":
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
                    # Aggregate responses from multiple setups
                    if not hasattr(self, 'dataset_names') or self.dataset_names is None:
                        self.dataset_names = []
                    
                    # Add new files, avoiding duplicates based on dataset_name
                    existing_names = {item.get('dataset_name', '') for item in self.dataset_names if isinstance(item, dict)}
                    for file_item in files_list:
                        if isinstance(file_item, dict):
                            dataset_name = file_item.get('dataset_name', '')
                            if dataset_name and dataset_name not in existing_names:
                                self.dataset_names.append(file_item)
                                existing_names.add(dataset_name)
                    
                    self.recv_dataset_names_status = True
                    self.ui_logger.info("Aggregated dataset files: %s", self.dataset_names)
                else:
                    self.ui_logger.warning("'report_dataset_names' response has no 'files' list.")
            # For "report_model_names" if the controller sends that
            elif message_cmd == "report_model_names":
                # The new format includes "files": [ {...}, {...} ]
                files_list = data.get("files", [])
                if isinstance(files_list, list):
                    # Aggregate responses from multiple setups
                    if not hasattr(self, 'model_names') or self.model_names is None:
                        self.model_names = []
                    
                    # Add new files, avoiding duplicates based on model_name
                    existing_names = {item.get('model_name', '') for item in self.model_names if isinstance(item, dict)}
                    for file_item in files_list:
                        if isinstance(file_item, dict):
                            model_name = file_item.get('model_name', '')
                            if model_name and model_name not in existing_names:
                                self.model_names.append(file_item)
                                existing_names.add(model_name)
                    
                    self.recv_model_names_status = True
                    self.ui_logger.info("Aggregated model files: %s", self.model_names)
                else:
                    self.ui_logger.warning("'report_model_names' response has no 'files' list.")
            else:
                # For all other commands, just log them
                self.ui_logger.info("Response for '%s' command: %s", message_cmd, data)
        else:
            self.ui_logger.warning("Unknown response type: %s", msg_type)

        # Acknowledge that we processed the message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def _add_error_message(self, error_msg):
        """Add an error message to the queue for frontend notification."""
        import time
        with self.error_lock:
            # Add timestamp and limit queue size to prevent memory issues
            error_entry = {
                'message': error_msg,
                'timestamp': time.time()
            }
            self.error_messages.append(error_entry)
            # Keep only the last 10 error messages
            if len(self.error_messages) > 10:
                self.error_messages = self.error_messages[-10:]

    def get_error_messages(self):
        """Get and clear all pending error messages."""
        with self.error_lock:
            messages = self.error_messages.copy()
            self.error_messages.clear()
            return messages


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

    def get_setup_info(self):
        """Extract setup information from config for UI rendering"""
        try:
            active_setups = self.config["hardware"].get("active_setups", [])
            if not active_setups:
                active_setups = ["setup_1"]  # Default fallback
                
            setup_info = []
            for setup_name in active_setups:
                try:
                    setup_data = self.config["hardware"][setup_name]
                    setup_id = str(setup_data.get('setup_id', '1'))
                    setup_info.append({
                        'name': setup_name,
                        'setup_id': setup_id,
                        'id': setup_id,
                        'display_name': f'Setup {setup_id}'
                    })
                except KeyError:
                    # Fallback based on setup name
                    setup_id = '1' if setup_name == "setup_1" else '2'
                    setup_info.append({
                        'name': setup_name,
                        'setup_id': setup_id,
                        'id': setup_id,
                        'display_name': f'Setup {setup_id}'
                    })
            
            return setup_info
        except (KeyError, TypeError) as e:
            self.ui_logger.error(f"Error reading setup info from config: {e}")
            # Complete fallback
            return [{'name': 'setup_1', 'setup_id': '1', 'id': '1'}]        

    def get_total_setups_count(self):
        """Get the total number of setups defined in the config (not just active ones)"""
        try:
            hardware_config = self.config.get("hardware", {})
            # Count setup_1, setup_2, etc. entries
            setup_count = 0
            for key in hardware_config.keys():
                if key.startswith("setup_") and key != "setup_id":  # Exclude any global setup_id if it exists
                    setup_count += 1
            return setup_count
        except Exception as e:
            self.ui_logger.error(f"Error counting total setups: {e}")
            return 1  # Fallback

    def get_available_setup_ids(self):
        """Get the setup IDs that are currently active/available"""
        setup_info = self.get_setup_info()
        return [setup['setup_id'] for setup in setup_info]



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

def ensure_controller_running(ui_logger, open_terminal, controller_path, rabbit_conf, config, setup_id=None):
    """
    Ensures controllers are running for ALL active setups.
    Launches separate controllers with --setup parameters for each active setup.
    This function is called for each send_command but ensures all controllers are running.
    """
    # Get active setups from config
    try:
        active_setups = config["hardware"].get("active_setups", [])
        if not active_setups:
            active_setups = ["setup_1", "setup_2"]  # Fallback
        
        setup_ids = []
        for setup_name in active_setups:
            try:
                setup_data = config["hardware"][setup_name]
                setup_ids.append(setup_data['setup_id'])
            except KeyError:
                # Fallback based on setup name
                if setup_name == "setup_1":
                    setup_ids.append(1)
                elif setup_name == "setup_2":
                    setup_ids.append(2)
    except (KeyError, TypeError):
        ui_logger.error("Error reading active setups from config. Using fallback setups.")
        setup_ids = [1]  # Complete fallback

    ui_logger.info(f"Ensuring controllers are running for all setups: {setup_ids}")
    
    # Check which controllers are already running by looking for --setup arguments
    running_setup_ids = set()
    try:
        result = subprocess.check_output(["pgrep", "-f", "robot_controller/main.py"], text=True)
        running_pids = result.strip().split('\n') if result.strip() else []
        
        for pid in running_pids:
            try:
                cmdline = subprocess.check_output(["ps", "-p", pid, "-o", "args", "--no-headers"], text=True).strip()
                # Look for --setup argument
                if "--setup" in cmdline:
                    parts = cmdline.split()
                    for i, part in enumerate(parts):
                        if part == "--setup" and i + 1 < len(parts):
                            try:
                                running_setup_id = int(parts[i + 1])
                                running_setup_ids.add(running_setup_id)
                                ui_logger.info(f"Found running controller for setup {running_setup_id} (PID: {pid})")
                            except ValueError:
                                pass
            except subprocess.CalledProcessError:
                pass
    except subprocess.CalledProcessError:
        ui_logger.info("No robot controllers found running.")
    
    # Launch controllers for missing setups (ALL AT ONCE)
    missing_setup_ids = [sid for sid in setup_ids if sid not in running_setup_ids]
    
    if missing_setup_ids:
        ui_logger.info(f"Launching controllers for missing setups: {missing_setup_ids}")
        
        # Check if we need ROS environment
        needs_ros = False
        try:
            # Check any setup for ROS requirement (assume same for all)
            first_setup = f"setup_{setup_ids[0]}"
            setup_config = config.get("hardware", {}).get(first_setup, {})
            robot_brand = setup_config.get("robot", {}).get("brand", "").lower()
            teachbot_brand = setup_config.get("teachbot", {}).get("brand", "").lower()
            
            if robot_brand == "interbotix" or teachbot_brand == "interbotix":
                needs_ros = True
                ui_logger.info("Interbotix hardware detected. Launching controllers with ROS environment.")
        except Exception as e:
            ui_logger.warning(f"Could not check hardware configuration: {e}. Launching without ROS.")
        
        # Launch all missing controllers simultaneously
        for missing_setup_id in missing_setup_ids:
            controller_cmd = f"python3 {controller_path} --setup {missing_setup_id}"
            
            if needs_ros:
                if open_terminal:
                    subprocess.Popen([
                        "gnome-terminal", "--window", "--title", f"Robot Controller Setup {missing_setup_id}", 
                        "--", "bash", "-c",
                        f"source /opt/ros/noetic/setup.bash && source $HOME/TOS/devel/setup.bash && {controller_cmd}; exec bash"
                    ])
                    ui_logger.info(f"Launched controller for setup {missing_setup_id} in new terminal with ROS.")
                else:
                    subprocess.Popen([
                        "bash", "-c", 
                        f"source /opt/ros/noetic/setup.bash && source $HOME/TOS/devel/setup.bash && {controller_cmd}"
                    ])
                    ui_logger.info(f"Launched controller for setup {missing_setup_id} in background with ROS.")
            else:
                if open_terminal:
                    subprocess.Popen([
                        "gnome-terminal", "--window", "--title", f"Robot Controller Setup {missing_setup_id}",
                        "--", "bash", "-c", f"{controller_cmd}; exec bash"
                    ])
                    ui_logger.info(f"Launched controller for setup {missing_setup_id} in new terminal.")
                else:
                    subprocess.Popen(["bash", "-c", controller_cmd])
                    ui_logger.info(f"Launched controller for setup {missing_setup_id} in background.")
        
        # Wait for all new controllers to initialize
        ui_logger.info(f"Waiting for {len(missing_setup_ids)} new controllers to initialize...")
        time.sleep(3)
    else:
        ui_logger.info("All required controllers are already running.")


   
