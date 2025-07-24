import os
import logging
import logging.config
import subprocess
import webbrowser
import yaml
import pika
import time
from pika.exceptions import AMQPConnectionError, AMQPChannelError
import struct, time
from multiprocessing import shared_memory
from typing import Iterator, List, Tuple


def calculate_shm_capacity(record_duration: float, control_dt: float) -> int:
    """
    Calculate shared memory buffer capacity using deterministic formula.
    This ensures Python and C++ use the same capacity regardless of config.
    
    Args:
        record_duration: Recording duration in seconds
        control_dt: Control timestep in seconds
        
    Returns:
        capacity: Buffer capacity (power of 2)
    """
    size = int(record_duration / control_dt)
    # Round size up to nearest power of 2 for efficient wraparound
    capacity = 2 ** (size - 1).bit_length()
    return capacity


def load_config(config_path="config.yaml"):
    """
    Loads the YAML config file.
    """
    # get the config from a directory back to the current file
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



def setup_logging(component_tag):
    # load config
    config = load_config()
    log_level = config["logging"]["log_level"]
    library_log_levels = config["logging"].get("library_log_levels", {})
    logging_config_dict = config["logging"]["config_dict"]

    # Convert main log level to numeric
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Ensure logs folder exists for log files
    log_directory = os.path.dirname(logging_config_dict["handlers"]["file"]["filename"])
    safe_tag = component_tag.replace("[", "").replace("]", "").replace(" ", "_").upper()
    # change file name from dummy_log to component_tag.log
    logging_config_dict["handlers"]["file"]["filename"] = os.path.join(log_directory, f"{safe_tag}.log")



    # Inject the component tag into the log format if "standard" formatter exists
    fmt_key = "standard"
    if ("formatters" in logging_config_dict 
        and fmt_key in logging_config_dict["formatters"] 
        and "format" in logging_config_dict["formatters"][fmt_key]):

        original_format = logging_config_dict["formatters"][fmt_key]["format"]
        # If the original format contains something like [%(levelname)s][...], replace "[...]" with component_tag:
        updated_format = original_format.replace("[...]", f"{component_tag}")
        logging_config_dict["formatters"][fmt_key]["format"] = updated_format

    # Update console + file handler levels
    if "handlers" in logging_config_dict:
        if "console" in logging_config_dict["handlers"]:
            logging_config_dict["handlers"]["console"]["level"] = log_level.upper()
        if "file" in logging_config_dict["handlers"]:
            logging_config_dict["handlers"]["file"]["level"] = log_level.upper()

    # Update root logger level
    if "root" in logging_config_dict:
        logging_config_dict["root"]["level"] = log_level.upper()

    # Apply the updated logging config
    logging.config.dictConfig(logging_config_dict)

    # Apply library-specific log levels if provided by config file     
    if library_log_levels:
        for lib_name, lib_level_str in library_log_levels.items():
            lvl = getattr(logging, lib_level_str.upper(), None)
            if isinstance(lvl, int):
                logging.getLogger(lib_name).setLevel(lvl)
            else:
                raise ValueError(f"Invalid library log level: {lib_level_str} for {lib_name}")

    return logging.getLogger(safe_tag)



def robust_connect(rabbit_conf, ui_logger=None, client_name=None):
    """
    Attempt to connect to RabbitMQ with parameters from 'rabbit_conf'.
    You can override 'client_name' if needed; otherwise it defaults
    to rabbit_conf["client_name_default"].
    """
    host = rabbit_conf["host"]
    user = rabbit_conf["user"]
    password = rabbit_conf["pass"]
    heartbeat = rabbit_conf["heartbeat"]
    blocked_connection_timeout = rabbit_conf["blocked_connection_timeout"]
    max_retries = rabbit_conf["max_retries"]
    wait_seconds = rabbit_conf["wait_seconds"]

    if client_name is None:
        client_name = rabbit_conf["client_name_default"]

    creds = pika.PlainCredentials(user, password)
    client_props = {
        "connection_name": client_name,
        "product": rabbit_conf["product"],
        "information": rabbit_conf["information"]
    }

    attempts = 0
    while True:
        try:
            params = pika.ConnectionParameters(
                host=host,
                credentials=creds,
                heartbeat=heartbeat,
                blocked_connection_timeout=blocked_connection_timeout,
                client_properties=client_props
            )
            conn = pika.BlockingConnection(params)
            return conn
        except (AMQPConnectionError, AMQPChannelError) as e:
            attempts += 1
            if ui_logger:
                ui_logger.warning(
                    f"[robust_connect] Connection failed: {e}. "
                    f"Retry {attempts}. Waiting {wait_seconds}s."
                )
            time.sleep(wait_seconds)
            if max_retries > 0 and attempts >= max_retries:
                raise


class RingBufferReader:
    def __init__(self, config=None, shm_key="shm_joint_data1", setup_id="1"):
        if config is None:
            config = load_config()
        
        # Read constants from config
        self.DOF_ROBOT = config["hardware"]["robot"]["dof"]  # Robot joints only  
        self.DOF_EE = config["hardware"]["robot"]["dof_ee"]   # End effector DOF (gripper)
        self.DOF = self.DOF_ROBOT + self.DOF_EE  # Total DOF including gripper
        cpp_config = config["cpp"]["shared_memory"][shm_key]
        self.SLOT_FMT = cpp_config["slot_format_template"].format(dof=self.DOF)  # Substitute DOF
        self.SLOT_SIZE = struct.calcsize(self.SLOT_FMT)
        self.HEADER_FMT = cpp_config["header_format"]
        self.HEADER_SIZE = struct.calcsize(self.HEADER_FMT)
        # Make shared memory name setup-specific
        self.SHM_NAME = f"{int(setup_id):02d}_{cpp_config['shm_name']}"
        
        self.shm = shared_memory.SharedMemory(name=self.SHM_NAME, create=False)
        self.buf  = self.shm.buf
        self.view = memoryview(self.buf)
        
        # Read the actual capacity from the shared memory header (set by C++ code)
        # Header format: write_idx, read_idx, capacity, slot_size
        header = struct.unpack_from(self.HEADER_FMT, self.view, 0)
        self.CAPACITY = header[2]  # capacity is the 3rd field in header
        actual_slot_size = header[3]  # slot_size is the 4th field in header
        
        # Verify that the slot size matches what we expect
        if actual_slot_size != self.SLOT_SIZE:
            import logging
            logger = logging.getLogger("RingBufferReader")
            logger.warning("Slot size mismatch: expected %d, got %d from header", 
                         self.SLOT_SIZE, actual_slot_size)
            # Use the size from the header to be safe
            self.SLOT_SIZE = actual_slot_size

    def close(self):
        """
        Close the memory view and shared memory connection.
        """
        try:
            if hasattr(self, 'view') and self.view is not None:
                # Clear all references to the memoryview
                del self.view
                self.view = None
        except Exception:
            pass
        
        try:
            if hasattr(self, 'buf'):
                # Clear buffer reference
                del self.buf
                self.buf = None
        except Exception:
            pass
        
        try:
            if hasattr(self, 'shm') and self.shm is not None:
                self.shm.close()
                self.shm = None
        except Exception:
            pass

    # ---------- header helpers ----------
    def _hdr(self) -> Tuple[int,int,int,int]:
        return struct.unpack_from(self.HEADER_FMT, self.view, 0)

    def _write_idx(self) -> int:
        return struct.unpack_from("I", self.view, 0)[0]

    def _read_idx(self) -> int:
        return struct.unpack_from("I", self.view, 4)[0]

    def _set_read_idx(self, idx:int) -> None:
        struct.pack_into("I", self.view, 4, idx)

    # ---------- public API ----------
    def __iter__(self):
        return self

    def __next__(self):
        while True:
            w = self._write_idx()
            r = self._read_idx()
            if r == w:                       # nothing new yet
                time.sleep(0.001)
                continue

            slot_off = self.HEADER_SIZE + (r & (self.CAPACITY - 1)) * self.SLOT_SIZE
            raw      = self.view[slot_off : slot_off + self.SLOT_SIZE]
            
            # Debug logging for unpack error
            if len(raw) != self.SLOT_SIZE:
                import logging
                logger = logging.getLogger("RingBufferReader")
                logger.error("Buffer size mismatch: expected %d bytes, got %d bytes", 
                           self.SLOT_SIZE, len(raw))
                logger.error("SLOT_FMT: %s", self.SLOT_FMT)
                logger.error("Expected struct size: %d", struct.calcsize(self.SLOT_FMT))
                logger.error("Shared memory details:")
                logger.error("  Total buffer size: %d bytes", len(self.view))
                logger.error("  HEADER_SIZE: %d bytes", self.HEADER_SIZE)
                logger.error("  CAPACITY: %d slots", self.CAPACITY)
                logger.error("  SLOT_SIZE: %d bytes", self.SLOT_SIZE)
                logger.error("  Expected total size: %d bytes", self.HEADER_SIZE + self.CAPACITY * self.SLOT_SIZE)
                logger.error("  Read index: %d, Write index: %d", r, w)
                logger.error("  Slot offset: %d", slot_off)
                logger.error("  Reading from %d to %d", slot_off, slot_off + self.SLOT_SIZE)
                raise ValueError(f"Buffer size mismatch: expected {self.SLOT_SIZE} bytes, got {len(raw)} bytes")
            
            data     = struct.unpack(self.SLOT_FMT, raw)
            self._set_read_idx((r + 1) & (self.CAPACITY - 1))

            # C++ structure uses kMaxDof=8 arrays with gripper state as 7th element
            kMaxDof = 8
            
            # Extract full position arrays (DOF_ROBOT + DOF_EE elements)
            # The gripper state is now included as the last element in each array
            total_dof = self.DOF_ROBOT + self.DOF_EE  # 6 + 1 = 7
            
            teachbot_position = list(data[0:total_dof])                      # 7 elements: 6 joints + gripper
            send_pos_robot = list(data[kMaxDof:kMaxDof + total_dof])        # 7 elements: 6 joints + gripper  
            robot_pos = list(data[2*kMaxDof:2*kMaxDof + total_dof])         # 7 elements: 6 joints + gripper
            
            return {
                "teachbot_position":     teachbot_position,                    # Robot joints + gripper state
                "sent_robot_position": send_pos_robot,                      # Robot joints + gripper state  
                "robot_position":      robot_pos,                           # Robot joints + gripper state
                "seq_id":              data[3*kMaxDof + 2],                 # seq_id uint32_t
                "timestamp":           data[3*kMaxDof + 3],                 # timestamp double
                "robot_position_timestamp": data[3*kMaxDof + 4],            # robot_position_timestamp double
            }
    
    def close(self, unlink: bool = False):
        self.shm.close()
        if unlink:
            self.shm.unlink()

    def read_available(self, max_entries=None):
        """
        Read all available entries without blocking.
        Returns a list of data entries.
        """
        entries = []
        count = 0
        
        while True:
            w = self._write_idx()
            r = self._read_idx()
            if r == w:  # nothing new
                break
                
            if max_entries and count >= max_entries:
                break
                
            slot_off = self.HEADER_SIZE + (r & (self.CAPACITY - 1)) * self.SLOT_SIZE
            raw      = self.view[slot_off : slot_off + self.SLOT_SIZE]
            data     = struct.unpack(self.SLOT_FMT, raw)
            self._set_read_idx((r + 1) & (self.CAPACITY - 1))

            # C++ structure uses kMaxDof=8 arrays with gripper state as 7th element
            kMaxDof = 8
            
            # Extract full position arrays (DOF_ROBOT + DOF_EE elements)
            # The gripper state is now included as the last element in each array
            total_dof = self.DOF_ROBOT + self.DOF_EE  # 6 + 1 = 7
            
            teachbot_position = list(data[0:total_dof])                      # 7 elements: 6 joints + gripper
            send_pos_robot = list(data[kMaxDof:kMaxDof + total_dof])        # 7 elements: 6 joints + gripper  
            robot_pos = list(data[2*kMaxDof:2*kMaxDof + total_dof])         # 7 elements: 6 joints + gripper
            
            entry = {
                "teachbot_position":     teachbot_position,                    # Robot joints + gripper state
                "sent_robot_position": send_pos_robot,                      # Robot joints + gripper state  
                "robot_position":      robot_pos,                           # Robot joints + gripper state
                "seq_id":              data[3*kMaxDof + 2],                 # seq_id uint32_t
                "timestamp":           data[3*kMaxDof + 3],                 # timestamp double
                "robot_position_timestamp": data[3*kMaxDof + 4],            # robot_position_timestamp double
            }
            entries.append(entry)
            count += 1
            
        return entries

def get_data_path(config, relative_path="", create_dirs=True):
    """
    Get the absolute path to the data directory, optionally with a relative path appended.
    Handles both relative and absolute data directory configurations.
    Optionally creates the directory if it doesn't exist.
    
    Args:
        config: The loaded configuration dictionary
        relative_path: Optional relative path to append to the data directory
        create_dirs: Whether to create directories if they don't exist (default: True)
        
    Returns:
        str: Absolute path to the data directory (optionally with relative_path appended)
    """
    app_directory = config["general"]["app_directory"]
    data_directory = config["general"]["data_directory"]
    
    # If data_directory is absolute, use it directly
    if os.path.isabs(data_directory):
        base_path = data_directory
    else:
        # If data_directory is relative, resolve it relative to app_directory
        if os.path.isabs(app_directory):
            base_path = os.path.join(app_directory, data_directory)
        else:
            # Both are relative, resolve from the current working directory
            base_path = os.path.abspath(os.path.join(app_directory, data_directory))
    
    # Append the relative path if provided
    if relative_path:
        full_path = os.path.join(base_path, relative_path)
    else:
        full_path = base_path
    
    # Create the directory if it doesn't exist and create_dirs is True
    if create_dirs:
        os.makedirs(full_path, exist_ok=True)
    
    return full_path