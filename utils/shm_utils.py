#!/usr/bin/env python3
"""
Shared memory utility functions for POSIX shared memory management.
These functions provide generic helpers for creating and managing shared memory
segments that can be reused across different components.
"""

import struct
from multiprocessing import shared_memory


def _initialize_single_shared_memory(shm_key, capacity, dof, logger, config, setup_id=None):
    """
    Initialize a single shared memory segment.
    Generic "create-or-recreate a POSIX shared-memory ring buffer" routine.
    
    Args:
        shm_key: Configuration key for the shared memory segment
        capacity: Number of slots in the buffer
        dof: Degrees of freedom for the data structure
        logger: Logger instance for logging events
        config: Configuration dictionary containing shared memory settings
        setup_id: Optional setup ID for multi-setup configurations
        
    Returns:
        None
        
    Raises:
        Exception: If shared memory creation fails after cleanup attempts
    """
    cpp_config = config["cpp"]["shared_memory"][shm_key]
    CAPACITY = capacity
    SLOT_FMT_TEMPLATE = cpp_config["slot_format_template"]
    SLOT_FMT = SLOT_FMT_TEMPLATE.format(dof=dof)  # Dynamically substitute total DOF
    SLOT_SIZE = struct.calcsize(SLOT_FMT)
    HEADER_FMT = cpp_config["header_format"]
    HEADER_SIZE = struct.calcsize(HEADER_FMT)
    
    # Make shared memory name setup-specific if setup_id provided
    if setup_id:
        SHM_NAME = f"{int(setup_id):02d}_{cpp_config['shm_name']}"
    else:
        SHM_NAME = cpp_config['shm_name']
    
    logger.info("Creating shared memory segment: %s with DOF=%d", SHM_NAME, dof)
    logger.info("SLOT_FMT_TEMPLATE: %s", SLOT_FMT_TEMPLATE)
    logger.info("SLOT_FMT: %s", SLOT_FMT)
    logger.info("SLOT_SIZE: %d bytes", SLOT_SIZE)
    logger.info("HEADER_SIZE: %d bytes", HEADER_SIZE)
    logger.info("CAPACITY: %d slots", CAPACITY)
    bytes_needed = HEADER_SIZE + CAPACITY * SLOT_SIZE
    logger.info("Total bytes needed: %d", bytes_needed)
    
    # Try to create shared memory, handling the case where it already exists
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=bytes_needed)
        # initialise header once
        struct.pack_into(HEADER_FMT, shm.buf, 0, 0, 0, CAPACITY, SLOT_SIZE)
        logger.info("Shared memory segment created with size: %d bytes", bytes_needed)
        shm.close()
    except FileExistsError:
        logger.warning("Shared memory segment already exists, cleaning up and recreating")
        try:
            # Try to attach to existing memory and clean it up
            existing_shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
            existing_shm.close()
            existing_shm.unlink()
            logger.info("Cleaned up existing shared memory segment")
            
            # Now create new shared memory
            shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=bytes_needed)
            struct.pack_into(HEADER_FMT, shm.buf, 0, 0, 0, CAPACITY, SLOT_SIZE)
            logger.info("Shared memory segment recreated with size: %d bytes", bytes_needed)
            shm.close()
        except Exception as cleanup_error:
            logger.error("Failed to clean up existing shared memory: %s", cleanup_error)
            raise


def _get_buffer_info_dict(buffer_dict):
    """
    Simple serializer turning a CameraRingBuffer dict into plain metadata.
    Converts a dict of CameraRingBuffer objects to a dict of buffer info dicts.
    
    Args:
        buffer_dict: Dictionary of CameraRingBuffer objects
        
    Returns:
        dict: Dictionary with same keys but buffer info dictionaries as values
    """
    return {
        name: {
            "name": buf.name,
            "width": buf.width,
            "height": buf.height,
            "channels": buf.channels,
            "capacity": buf.capacity
        }
        for name, buf in buffer_dict.items()
    }
