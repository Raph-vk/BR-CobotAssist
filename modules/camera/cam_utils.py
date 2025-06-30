#!/usr/bin/env python3

import struct
import time
import numpy as np
from multiprocessing import shared_memory
from typing import Optional, Dict, Any, List
import os

class CameraRingBuffer:
    """
    Shared memory ring buffer for camera images.
    Supports both color (RGB) and depth (16-bit) images with timestamps.
    
    Memory layout:
    [Header: write_idx(4) + read_idx(4) + capacity(4) + slot_size(4)] 
    [Slot 0: timestamp(8) + width(4) + height(4) + channels(4) + image_data]
    [Slot 1: ...]
    [Slot N-1: ...]
    """
    
    HEADER_FORMAT = "IIII"  # write_idx, read_idx, capacity, slot_size
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    
    SLOT_HEADER_FORMAT = "dIII"  # timestamp, width, height, channels
    SLOT_HEADER_SIZE = struct.calcsize(SLOT_HEADER_FORMAT)
    
    def __init__(self, name: str, width: int, height: int, channels: int, 
                 capacity: int, create: bool = True):
        """
        Initialize camera ring buffer.
        
        Args:
            name: Shared memory name
            width: Image width
            height: Image height  
            channels: Number of channels (3 for RGB, 1 for depth)
            capacity: Number of image slots (must be power of 2)
            create: Whether to create new shared memory or attach to existing
        """
        self.name = name
        self.width = width
        self.height = height
        self.channels = channels
        self.capacity = capacity
               
        # Calculate sizes
        self.image_data_size = width * height * channels
        if channels == 1:  # Depth images use 16-bit values
            self.image_data_size *= 2
            
        self.slot_size = self.SLOT_HEADER_SIZE + self.image_data_size
        self.total_size = self.HEADER_SIZE + (capacity * self.slot_size)
        
        # Create or attach to shared memory
        try:
            if create:
                # Validate capacity is power of 2 and > 0
                if capacity <= 0 or not (capacity & (capacity - 1)) == 0:
                    raise ValueError(f"Capacity must be a positive power of 2, got {capacity}")
                
                # Check if shared memory with same name exists and delete it
                try:
                    existing_shm = shared_memory.SharedMemory(name=name, create=False)
                    existing_shm.unlink()  # Delete the existing shared memory
                    existing_shm.close()
                except FileNotFoundError:
                    pass  # No existing shared memory, which is fine
                except Exception:
                    pass  # Ignore other errors during cleanup
                
                # Create new shared memory buffer
                self.shm = shared_memory.SharedMemory(name=name, create=True, size=self.total_size)
                self.buf = self.shm.buf
                self.view = memoryview(self.buf)
                self._initialize_header()
            else:
                self.shm = shared_memory.SharedMemory(name=name, create=False)
                self.buf = self.shm.buf
                self.view = memoryview(self.buf)
                # Read capacity from existing buffer header
                _, _, header_capacity, header_slot_size = self._get_header()
                
                # If capacity was 0 (placeholder), use the one from header
                if self.capacity == 0:
                    self.capacity = header_capacity
                    # Recalculate sizes with the correct capacity
                    self.slot_size = self.SLOT_HEADER_SIZE + self.image_data_size
                    self.total_size = self.HEADER_SIZE + (self.capacity * self.slot_size)
                    
                    # Validate the capacity is power of 2
                    if self.capacity <= 0 or not (self.capacity & (self.capacity - 1)) == 0:
                        raise ValueError(f"Existing buffer has invalid capacity: {self.capacity}")
                
                # Validate that existing buffer matches our requirements
                if header_capacity == self.capacity and header_slot_size == self.slot_size:
                    pass
                else:
                    # Buffer size mismatch - this shouldn't happen in normal operation
                    self._initialize_header()
        except Exception as e:
            # Handle shared memory creation errors
            if create:
                raise RuntimeError(f"Failed to create shared memory '{name}': {e}")
            else:
                raise RuntimeError(f"Failed to attach to shared memory '{name}': {e}")

        
    def _initialize_header(self):
        """Initialize header with zeros."""
        struct.pack_into(self.HEADER_FORMAT, self.view, 0, 0, 0, self.capacity, self.slot_size)
    
    def _get_header(self):
        """Get header values."""
        return struct.unpack_from(self.HEADER_FORMAT, self.view, 0)
    
    def _get_write_idx(self) -> int:
        """Get current write index, ensuring it's within bounds."""
        idx = struct.unpack_from("I", self.view, 0)[0]
        return idx & (self.capacity - 1)  # Ensure it's within bounds
    
    def _get_read_idx(self) -> int:
        """Get current read index, ensuring it's within bounds."""
        idx = struct.unpack_from("I", self.view, 4)[0]
        return idx & (self.capacity - 1)  # Ensure it's within bounds
    
    def _set_write_idx(self, idx: int):
        """Set write index, ensuring it's within bounds."""
        bounded_idx = idx & (self.capacity - 1)
        struct.pack_into("I", self.view, 0, bounded_idx)
    
    def _set_read_idx(self, idx: int):
        """Set read index, ensuring it's within bounds.""" 
        bounded_idx = idx & (self.capacity - 1)
        struct.pack_into("I", self.view, 4, bounded_idx)
    
    def write(self, image: np.ndarray, timestamp: float, camera_name: str = "", 
              serial_number: str = "", frame_type: str = "") -> bool:
        """
        Write image data to ring buffer.
        Always overwrites the current write position, handling circular buffer behavior.
        When buffer is full, the oldest images are automatically overwritten.
        
        Args:
            image: Image data as numpy array
            timestamp: Timestamp as float
            camera_name: Camera name (for logging)
            serial_number: Camera serial number (for logging) 
            frame_type: Frame type (color/depth)
            
        Returns:
            Always True (writes will always succeed by overwriting old data)
        """
        write_idx = self._get_write_idx()
        read_idx = self._get_read_idx()
        
        # Ensure write_idx is within bounds (fix any corrupted indices)
        write_idx = write_idx & (self.capacity - 1)
        read_idx = read_idx & (self.capacity - 1)
        
        # Calculate next write position using circular buffer logic
        next_write = (write_idx + 1) & (self.capacity - 1)
        
        # Check if this write will overwrite unread data
        if self.capacity > 1:
            # Standard ring buffer: check if next write would catch up to read pointer
            if next_write == read_idx:
                # Buffer is full - advance read pointer to discard oldest entry
                new_read_idx = (read_idx + 1) & (self.capacity - 1)
                self._set_read_idx(new_read_idx)
        # For capacity = 1, we always overwrite the single slot
            
        # Calculate slot offset for current write position
        slot_offset = self.HEADER_SIZE + (write_idx * self.slot_size)
        
        # Validate slot offset is within buffer bounds
        if slot_offset + self.slot_size > self.total_size:
            raise ValueError(f"Slot offset {slot_offset} + slot size {self.slot_size} exceeds buffer size {self.total_size}. write_idx={write_idx}, capacity={self.capacity}")
        
        # Validate image dimensions
        if len(image.shape) == 3:
            h, w, c = image.shape
        else:  # Depth image
            h, w = image.shape
            c = 1
            
        if w != self.width or h != self.height or c != self.channels:
            raise ValueError(f"Image dimensions {(h,w,c)} don't match buffer {(self.height,self.width,self.channels)}")
        
        # Write slot header with timestamp and dimensions
        struct.pack_into(self.SLOT_HEADER_FORMAT, self.view, slot_offset, 
                        timestamp, w, h, c)
        
        # Write image data
        image_offset = slot_offset + self.SLOT_HEADER_SIZE
        
        # Handle different data types and ensure proper byte conversion
        if image.dtype == np.uint8:
            image_bytes = image.tobytes()
        elif image.dtype == np.uint16:
            image_bytes = image.tobytes()
        else:
            # Convert to appropriate type based on buffer configuration
            if self.channels == 1:  # Depth images use 16-bit
                image_bytes = image.astype(np.uint16).tobytes()
            else:  # Color images use 8-bit
                image_bytes = image.astype(np.uint8).tobytes()
                
        # Write image data to shared memory buffer
        if len(image_bytes) != self.image_data_size:
            raise ValueError(f"Image data size {len(image_bytes)} doesn't match expected {self.image_data_size}")
            
        # Validate image data will fit in buffer
        if image_offset + len(image_bytes) > self.total_size:
            raise ValueError(f"Image data at offset {image_offset} + size {len(image_bytes)} exceeds buffer size {self.total_size}")
            
        self.view[image_offset:image_offset + len(image_bytes)] = image_bytes
        
        # Always advance write index to next position (circular) and ensure it's bounded
        self._set_write_idx(next_write)
        return True
        

    
    def read(self) -> Optional[Dict[str, Any]]:
        """
        Read next image from ring buffer.
        
        For capacity=1 buffers (policy buffers), always reads the latest data without 
        advancing read index, since data is overwritten each frame.
        For larger buffers, uses standard ring buffer logic with read/write indices.
        
        Returns:
            Dictionary with image data and metadata, or None if buffer empty
        """
        write_idx = self._get_write_idx()
        read_idx = self._get_read_idx()
        
        # Special handling for capacity=1 buffers (policy buffers)
        if self.capacity == 1:
            # Read from slot 0 (the only slot)
            slot_offset = self.HEADER_SIZE
            
            # Read slot header to check if valid data exists
            timestamp, width, height, channels = struct.unpack_from(
                self.SLOT_HEADER_FORMAT, self.view, slot_offset)
            
            # If timestamp is 0, no valid data has been written yet
            if timestamp == 0:
                return None
            
            # Read image data with proper memory management to avoid BufferError
            image_offset = slot_offset + self.SLOT_HEADER_SIZE
            
            # Create a completely independent copy to avoid shared memory references
            if channels == 1:  # Depth
                # Read directly into a new array to avoid intermediate references
                temp_view = self.view[image_offset:image_offset + self.image_data_size]
                image = np.frombuffer(temp_view, dtype=np.uint16).reshape((height, width)).copy()
                del temp_view  # Explicitly delete reference
            else:  # Color
                # Read directly into a new array to avoid intermediate references  
                temp_view = self.view[image_offset:image_offset + self.image_data_size]
                image = np.frombuffer(temp_view, dtype=np.uint8).reshape((height, width, channels)).copy()
                del temp_view  # Explicitly delete reference
            
            # Don't advance read index for capacity=1 buffers - always read the latest
            return {
                "image": image,
                "timestamp": timestamp,
                "frame_type": "depth" if channels == 1 else "color"
            }
        
        # Standard ring buffer logic for capacity > 1
        if read_idx == write_idx:
            return None
            
        # Calculate slot offset
        slot_offset = self.HEADER_SIZE + (read_idx * self.slot_size)
        
        # Read slot header
        timestamp, width, height, channels = struct.unpack_from(
            self.SLOT_HEADER_FORMAT, self.view, slot_offset)
        
        # Read image data with proper memory management to avoid BufferError
        image_offset = slot_offset + self.SLOT_HEADER_SIZE
        
        # Create a completely independent copy to avoid shared memory references
        if channels == 1:  # Depth
            # Read directly into a new array to avoid intermediate references
            temp_view = self.view[image_offset:image_offset + self.image_data_size]
            image = np.frombuffer(temp_view, dtype=np.uint16).reshape((height, width)).copy()
            del temp_view  # Explicitly delete reference
        else:  # Color
            # Read directly into a new array to avoid intermediate references
            temp_view = self.view[image_offset:image_offset + self.image_data_size]
            image = np.frombuffer(temp_view, dtype=np.uint8).reshape((height, width, channels)).copy()
            del temp_view  # Explicitly delete reference
        
        # Update read index for multi-capacity buffers
        next_read = (read_idx + 1) % self.capacity
        self._set_read_idx(next_read)
        
        return {
            "image": image,
            "timestamp": timestamp,
            "frame_type": "depth" if channels == 1 else "color"
        }
    
    def read_all_within_range(self, min_timestamp: float, max_timestamp: float) -> List[Dict[str, Any]]:
        """
        Read all images within timestamp range without modifying read pointer.
        
        Args:
            min_timestamp: Minimum timestamp
            max_timestamp: Maximum timestamp
            
        Returns:
            List of image data dictionaries
        """
        results = []
        write_idx = self._get_write_idx()
        read_idx = self._get_read_idx()
        
        # Check if buffer is empty
        if read_idx == write_idx:
            return results
            
        # Calculate how many items are available in the buffer
        available_count = (write_idx - read_idx) & (self.capacity - 1)
        
        # Start from the oldest available data
        current_idx = read_idx
        items_checked = 0
        
        while items_checked < available_count:
            slot_offset = self.HEADER_SIZE + (current_idx * self.slot_size)
            
            # Read just the timestamp first
            timestamp = struct.unpack_from("d", self.view, slot_offset)[0]

            # Check if timestamp is in range
            if min_timestamp <= timestamp <= max_timestamp:
                # Read full slot header
                timestamp, width, height, channels = struct.unpack_from(
                    self.SLOT_HEADER_FORMAT, self.view, slot_offset)
                
                # Read image data
                image_offset = slot_offset + self.SLOT_HEADER_SIZE
                image_bytes = bytes(self.view[image_offset:image_offset + self.image_data_size])
                
                # Convert bytes back to numpy array (copy data to avoid shared memory references)
                if channels == 1:  # Depth
                    image = np.frombuffer(image_bytes, dtype=np.uint16).reshape((height, width)).copy()
                else:  # Color
                    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, channels)).copy()
                
                results.append({
                    "image": image.copy(),  # Make copy since we're not consuming
                    "timestamp": timestamp,
                    "frame_type": "depth" if channels == 1 else "color"
                })
                
            current_idx = (current_idx + 1) & (self.capacity - 1)
            items_checked += 1
        
        return results
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        write_idx = self._get_write_idx()
        read_idx = self._get_read_idx()
        return read_idx == write_idx
    
    def is_full(self) -> bool:
        """
        Check if buffer is at capacity (next write would overwrite oldest data).
        Note: With circular buffer behavior, this doesn't prevent writes.
        """
        write_idx = self._get_write_idx()
        read_idx = self._get_read_idx()
        next_write = (write_idx + 1) & (self.capacity - 1)
        return next_write == read_idx
    
    def get_count(self) -> int:
        """Get number of items in buffer."""
        write_idx = self._get_write_idx()
        read_idx = self._get_read_idx()
        return (write_idx - read_idx) & (self.capacity - 1)
    
    def clear(self):
        """Clear buffer by setting read index to write index."""
        write_idx = self._get_write_idx()
        self._set_read_idx(write_idx)
    
    def close(self, unlink: bool = False):
        """Close shared memory properly."""
        try:
            # Close the memory view first to release references
            if hasattr(self, 'view') and self.view:
                self.view.release()
                self.view = None
        except Exception:
            pass
        
        try:
            # Clear buffer reference
            if hasattr(self, 'buf'):
                self.buf = None
        except Exception:
            pass
        
        try:
            # Close the shared memory
            if hasattr(self, 'shm') and self.shm:
                self.shm.close()
                
            if unlink and hasattr(self, 'shm') and self.shm:
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass  # Already unlinked
                except Exception:
                    pass
        except Exception:
            pass
    
    def empty(self) -> bool:
        """Compatibility method for queue interface."""
        return self.is_empty()
    
    def full(self) -> bool:
        """
        Compatibility method for queue interface.
        Note: With circular buffer behavior, buffer accepts writes even when "full".
        """
        return self.is_full()
    
    def get_nowait(self) -> Optional[Dict[str, Any]]:
        """Compatibility method for queue interface - non-blocking read."""
        result = self.read()
        if result is None:
            raise Exception("Ring buffer is empty")  # Similar to queue.Empty
        
        # Convert to format expected by existing code
        return {
            "image": result["image"],
            "timestamp": result["timestamp"],
            "camera_name": "",  # Will be filled by caller context
            "serial_number": "",  # Will be filled by caller context
            "frame_type": result["frame_type"]
        }
    
    def put_nowait(self, data: Dict[str, Any]) -> bool:
        """
        Compatibility method for queue interface - non-blocking write.
        Always succeeds with circular buffer behavior.
        """
        if isinstance(data, dict) and "image" in data and "timestamp" in data:
            return self.write(
                image=data["image"],
                timestamp=data["timestamp"],
                camera_name=data.get("camera_name", ""),
                serial_number=data.get("serial_number", ""),
                frame_type=data.get("frame_type", "")
            )
        return False
    



class CameraRingBufferManager:
    """
    Manages multiple camera ring buffers.
    """
    
    def __init__(self):
        self.color_buffers1 = {}
        self.depth_buffers1 = {}
        self.color_buffers2 = {}
        self.depth_buffers2 = {}
    
    def create_buffers(self, camera_configs: List[Dict], record_duration: int = 1, policy_img_buffer_size: int = 1) -> Dict[str, Any]:
        """
        Create ring buffers for all cameras.
        
        Args:
            camera_configs: List of camera configuration dictionaries
            record_duration: Recording duration in seconds for buffers1
            policy_img_buffer_size: Buffer size for policy interface buffers2
            
        Returns:
            Dictionary with buffer references
        """
        buffers_info = {
            "color_buffers1": {},
            "depth_buffers1": {},
            "color_buffers2": {},
            "depth_buffers2": {}
        }
        
        for camera_config in camera_configs:
            name = camera_config["name"]
            fps = camera_config["fps"]
            
            # Calculate buffer capacity for recording buffers (power of 2)
            desired_capacity = fps * record_duration
            capacity1 = 1
            while capacity1 < desired_capacity * 2:  # Extra headroom
                capacity1 *= 2

            # Policy buffers have a fixed small capacity
            capacity2 = policy_img_buffer_size
            
            # Color buffer 1 (for recording)
            color_name1 = f"camera_color1_{name}"
            color_buffer1 = CameraRingBuffer(
                name=color_name1,
                width=camera_config["color_width"],
                height=camera_config["color_height"],
                channels=3,
                capacity=capacity1,
                create=True
            )
            self.color_buffers1[name] = color_buffer1
            buffers_info["color_buffers1"][name] = color_buffer1
            
            # Color buffer 2 (for policy)
            color_name2 = f"camera_color2_{name}"
            color_buffer2 = CameraRingBuffer(
                name=color_name2,
                width=camera_config["color_width"],
                height=camera_config["color_height"],
                channels=3,
                capacity=capacity2,
                create=True
            )
            self.color_buffers2[name] = color_buffer2
            buffers_info["color_buffers2"][name] = color_buffer2
            
            # Depth buffer 1 (for recording)
            depth_name1 = f"camera_depth1_{name}"
            depth_buffer1 = CameraRingBuffer(
                name=depth_name1,
                width=camera_config["depth_width"],
                height=camera_config["depth_height"], 
                channels=1,
                capacity=capacity1,
                create=True
            )
            self.depth_buffers1[name] = depth_buffer1
            buffers_info["depth_buffers1"][name] = depth_buffer1
            
            # Depth buffer 2 (for policy)
            depth_name2 = f"camera_depth2_{name}"
            depth_buffer2 = CameraRingBuffer(
                name=depth_name2,
                width=camera_config["depth_width"],
                height=camera_config["depth_height"], 
                channels=1,
                capacity=capacity2,
                create=True
            )
            self.depth_buffers2[name] = depth_buffer2
            buffers_info["depth_buffers2"][name] = depth_buffer2
            
        return buffers_info
    
    def close_all(self, unlink: bool = False):
        """Close all buffers."""
        for buffer in self.color_buffers1.values():
            buffer.close(unlink=unlink)
        for buffer in self.depth_buffers1.values():
            buffer.close(unlink=unlink)
        for buffer in self.color_buffers2.values():
            buffer.close(unlink=unlink)
        for buffer in self.depth_buffers2.values():
            buffer.close(unlink=unlink)
        
        self.color_buffers1.clear()
        self.depth_buffers1.clear()
        self.color_buffers2.clear()
        self.depth_buffers2.clear()
    

