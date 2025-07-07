# TOS Application Shared Memory Documentation

This document describes all shared memory implementations used in the TOS (Teaching Operating System) application, including their communication patterns, data structures, synchronization mechanisms, and locking strategies.

## Overview

The TOS application uses multiple shared memory segments for inter-process communication between different components:

1. **C++ Ring Buffer Shared Memory** (Robot Control)
2. **Policy Target Shared Memory** (Policy Interface)
3. **Camera Ring Buffer Shared Memory** (Image Data)
4. **ACT-ReSiP Shared Memory** (Reinforcement Learning)

## 1. C++ Ring Buffer Shared Memory

### Purpose
High-performance data sharing between C++ control loops and Python interfaces for robot joint data.

### Communication Pattern
- **Between**: C++ control loop (writer) ↔ Python interfaces (readers)
- **How**: Lock-free SPSC (Single Producer, Single Consumer) ring buffer
- **Why**: Real-time robot control requires low-latency, high-frequency data exchange

### Memory Segments

#### 1.1 shm_cpp_joint_data1 (Save Interface)
```yaml
Name: "shm_cpp_joint_data1"
Purpose: Recording and data logging
Capacity: Runtime calculated (record_duration / control_dt, rounded to power of 2)
```

#### 1.2 shm_cpp_joint_data2 (Policy Interface)  
```yaml
Name: "shm_cpp_joint_data2"
Purpose: Real-time policy control
Capacity: 100 (fixed, configured in config.yaml)
```

### Data Structure

#### Header (RingBufferHeader)
```cpp
struct RingBufferHeader {
    std::atomic<uint32_t> write_idx{0};  // Current write position
    std::atomic<uint32_t> read_idx{0};   // Current read position  
    uint32_t capacity{0};                // Buffer capacity (power of 2)
    uint32_t slot_size{sizeof(JointDataSlot)}; // Size of each slot
};
```

#### Data Slot (JointDataSlot)
```cpp
struct JointDataSlot {
    double  master_pos[8];              // Target positions (kMaxDof=8)
    double  send_position_robot[8];     // Positions sent to robot
    double  robot_pos[8];               // Actual robot positions
    uint8_t gripper_on;                 // Gripper open command
    uint8_t gripper_off;                // Gripper close command  
    uint32_t seq_id;                    // Sequence ID
    double  timestamp;                  // Sample timestamp
    double  robot_position_timestamp;   // Robot position timestamp
};
```

#### Format String
```python
slot_format_template: "=8d8d8d??Idd"  # 8 doubles + 8 doubles + 8 doubles + 2 bytes + uint32 + 2 doubles
header_format: "=IIII"                # 4 uint32s
```

### Memory Layout
```
[Header: 16 bytes]
├── write_idx (4 bytes, atomic)
├── read_idx (4 bytes, atomic) 
├── capacity (4 bytes)
└── slot_size (4 bytes)

[Slot 0: sizeof(JointDataSlot) bytes]
[Slot 1: sizeof(JointDataSlot) bytes]
...
[Slot N-1: sizeof(JointDataSlot) bytes]
```

### Synchronization
- **Lock Type**: **Lock-free** using atomic operations
- **Mechanism**: Atomic compare-and-swap on write_idx/read_idx
- **Memory Ordering**: 
  - `memory_order_acquire` for reads
  - `memory_order_release` for writes
  - `memory_order_relaxed` for index checks

### Implementation Details
- **C++ Writer**: `ShmRingBufferWriter` class uses `mmap()` and `shm_open()`
- **Python Reader**: `RingBufferReader` class uses `multiprocessing.shared_memory`
- **Overflow Handling**: Circular buffer - oldest data is overwritten when full
- **Capacity**: Must be power of 2 for efficient wraparound using bitwise AND

---

## 2. Policy Target Shared Memory

### Purpose
Communication channel for policy interface to send target positions to robot control.

### Communication Pattern
- **Between**: Policy Interface (writer) ↔ Robot Control (reader)
- **How**: Sequential write buffer with sequence IDs
- **Why**: Policy needs to send continuous target positions for robot execution

### Configuration
```yaml
Name: "shm_target_pos2"
Capacity: 500 entries
Format: "=I{dof}d"  # sequence_id (uint32) + joint positions (double each)
```

### Data Structure

#### Entry Format
```python
# Template: "=I{dof}d" where {dof} is substituted at runtime
# Example for 7 DOF (6 joints + 1 gripper): "=I7d"
struct Entry {
    uint32_t sequence_id;     // Incremental sequence ID
    double positions[DOF];    // Joint positions (including gripper)
};
```

### Memory Layout
```
[Entry 0: 4 + DOF*8 bytes]
├── sequence_id (4 bytes)
└── positions (DOF * 8 bytes)

[Entry 1: 4 + DOF*8 bytes]
...
[Entry N-1: 4 + DOF*8 bytes]
```

### Synchronization
- **Lock Type**: **No explicit locking**
- **Mechanism**: Sequential writes with sequence IDs for ordering
- **Access Pattern**: Writer appends new entries, reader scans for latest

### Implementation Details
- **Writer**: Policy interfaces use `struct.pack()` to write binary data
- **Reader**: C++ control loop uses `mmap()` with read-only access
- **Size Calculation**: `capacity * (4 + DOF * 8)` bytes
- **Initialization**: Zero-filled on creation

---

## 3. Camera Ring Buffer Shared Memory

### Purpose
High-bandwidth image data sharing between camera processes and vision consumers.

### Communication Pattern
- **Between**: Camera processes (writers) ↔ Vision processing (readers)
- **How**: Ring buffer per camera with timestamp-based ordering
- **Why**: High-resolution images require efficient memory sharing to avoid copying

### Memory Segments
- **color_buffers1**: Color images from camera set 1
- **depth_buffers1**: Depth images from camera set 1  
- **color_buffers2**: Color images from camera set 2
- **depth_buffers2**: Depth images from camera set 2

### Data Structure

#### Header
```python
HEADER_FORMAT = "IIII"  # write_idx, read_idx, capacity, slot_size
```

#### Slot Structure
```python
SLOT_HEADER_FORMAT = "dIII"  # timestamp, width, height, channels

struct ImageSlot {
    double timestamp;      // Image capture timestamp
    uint32_t width;        // Image width in pixels
    uint32_t height;       // Image height in pixels  
    uint32_t channels;     // Number of channels (3=RGB, 1=depth)
    uint8_t image_data[];  // Raw image data
};
```

### Memory Layout
```
[Header: 16 bytes]
├── write_idx (4 bytes)
├── read_idx (4 bytes)
├── capacity (4 bytes) 
└── slot_size (4 bytes)

[Slot 0]
├── timestamp (8 bytes)
├── width (4 bytes)
├── height (4 bytes)
├── channels (4 bytes)
└── image_data (width * height * channels * bytes_per_pixel)

[Slot 1]
...
[Slot N-1]
```

### Synchronization
- **Lock Type**: **No explicit locking**
- **Mechanism**: Ring buffer with atomic index updates
- **Overflow**: Circular overwrite - newest images replace oldest
- **Validation**: Capacity must be power of 2

### Implementation Details
- **Image Types**: RGB (3 channels, 1 byte/pixel) or Depth (1 channel, 2 bytes/pixel)
- **Buffer Management**: `CameraRingBufferManager` handles multiple cameras
- **Memory Safety**: Bounds checking using bitwise AND with (capacity-1)

---

## 4. ACT-ReSiP Shared Memory (Reinforcement Learning)

### Purpose
Multi-process reinforcement learning with shared state between detector, policy, and runner processes.

### Communication Pattern
- **Between**: ReSiP Process ↔ Detectron Process ↔ Runner Process
- **How**: Multiple shared arrays with explicit locking
- **Why**: RL requires synchronized access to shared state and flags

### Memory Segments

#### Data Arrays
```python
qpos_shm:                  # Robot joint positions
shape: rl_cfg.qpos_shape
dtype: np.float32

vision_data_shm:           # Camera vision data  
shape: (num_cameras, *vision_shape)
dtype: np.float32

next_base_action_shm:      # Next action from policy
shape: rl_cfg.action_shape  
dtype: np.float32

final_action_shm:          # Final processed action
shape: rl_cfg.action_shape
dtype: np.float32
```

#### Control Flags
```python
predict_action_flag_shm:    # Trigger action prediction
shape: rl_cfg.flag_shape
dtype: np.uint8

action_ready_flag_shm:      # Action ready signal
shape: rl_cfg.flag_shape  
dtype: np.uint8

activate_detectron_flag_shm: # Enable object detection
shape: rl_cfg.flag_shape
dtype: np.uint8

reward_flag_shm:           # Reward signal
shape: rl_cfg.flag_shape
dtype: np.uint8

env_reset_flag_shm:        # Environment reset trigger
shape: rl_cfg.flag_shape
dtype: np.uint8
```

### Data Structure
```python
# Each shared memory segment is a raw numpy array
# accessed via: np.ndarray(shape, dtype=dtype, buffer=shm.buf)
```

### Synchronization
- **Lock Type**: **Explicit Python multiprocessing.Lock()**
- **Usage**: All shared memory access wrapped in `with shm_lock:` blocks
- **Scope**: Single lock protects all shared memory segments
- **Granularity**: Coarse-grained locking for simplicity

### Implementation Example
```python
# Reading with lock
with shm_lock:
    action_ready = np.ndarray(rl_cfg.flag_shape, dtype=np.uint8, 
                             buffer=action_ready_flag_shm.buf)[0]

# Writing with lock  
with shm_lock:
    np.ndarray(rl_cfg.flag_shape, dtype=np.uint8, 
               buffer=predict_action_flag_shm.buf)[0] = 1
```

### Process Communication Flow
```
Runner Process:
1. Check env_reset_flag with lock
2. If action_ready_flag set, read final_action with lock
3. Clear action_ready_flag with lock

ReSiP Process:
1. Wait for predict_action_flag with lock
2. Read qpos and vision_data with lock
3. Compute action and write to final_action_shm with lock
4. Set action_ready_flag with lock

Detectron Process:
1. Wait for activate_detectron_flag with lock
2. Process detection and set reward_flag with lock
```

---

## Summary of Locking Mechanisms

| Shared Memory Type | Lock Mechanism | Lock Granularity | Performance Characteristics |
|-------------------|----------------|------------------|---------------------------|
| C++ Ring Buffer | **Lock-free atomic** | Per-operation | Highest performance, real-time capable |
| Policy Target | **No locking** | N/A | High performance, eventual consistency |
| Camera Ring Buffer | **No locking** | N/A | High bandwidth, overwrite on overflow |
| ACT-ReSiP | **Explicit mutex** | Coarse-grained | Lower performance, strong consistency |

## Key Design Principles

1. **Lock-free for Real-time**: Critical robot control paths use lock-free atomic operations
2. **Explicit Locking for Complex State**: Multi-flag RL systems use explicit locks for consistency
3. **No Locking for High Bandwidth**: Image data uses overwrite semantics instead of blocking
4. **Power-of-2 Capacities**: Enable efficient wraparound using bitwise operations
5. **Sequence IDs**: Provide ordering and duplicate detection without locks

## Memory Management

- **Creation**: Shared memory segments created by controller/main processes
- **Cleanup**: Explicit cleanup on shutdown with `unlink()` to remove segments
- **Error Handling**: Graceful handling of existing segments and creation failures
- **Platform**: Uses POSIX shared memory (`shm_open`/`mmap`) on Linux

---

## Atomic Ring Buffer Analysis and Recommendations

### Current State Analysis

You raise an excellent point about standardizing on atomic ring buffers! Let's analyze the current implementations:

#### C++ Ring Buffer (BEST IMPLEMENTATION)
```cpp
struct RingBufferHeader {
    std::atomic<uint32_t> write_idx{0};  // Proper atomic with memory ordering
    std::atomic<uint32_t> read_idx{0};   // Proper atomic with memory ordering
    uint32_t capacity{0};                // Static after initialization
    uint32_t slot_size{0};              // Static after initialization
};

void push(const JointDataSlot& s, int dof) {
    uint32_t w = hdr_->write_idx.load(std::memory_order_relaxed);
    uint32_t r = hdr_->read_idx.load(std::memory_order_acquire);    // 🔑 Key: acquire semantics
    uint32_t nxt = (w + 1) & (capacity_ - 1);
    
    if (nxt == r) { // Buffer full - drop oldest
        hdr_->read_idx.store((r + 1) & (capacity_ - 1), std::memory_order_release);
    }
    
    // Write data to slot
    std::memcpy(&slots_[w], &s, sizeof(JointDataSlot));
    hdr_->write_idx.store(nxt, std::memory_order_release);         // 🔑 Key: release semantics
}
```

#### Camera Ring Buffer (PROBLEMATIC - NOT ATOMIC)
```python
def _set_write_idx(self, idx: int):
    bounded_idx = idx & (self.capacity - 1)
    struct.pack_into("I", self.view, 0, bounded_idx)  # ❌ NOT ATOMIC!

def _set_read_idx(self, idx: int):
    bounded_idx = idx & (self.capacity - 1) 
    struct.pack_into("I", self.view, 4, bounded_idx)  # ❌ NOT ATOMIC!
```

### Memory Ordering Guarantees

#### What Atomic Indices Guarantee:

1. **No Torn Reads/Writes**: Atomic ensures the 32-bit index is read/written as a single operation
2. **Memory Ordering**: `acquire/release` creates synchronization points
3. **Visibility**: Changes become visible to other processes in a predictable order

#### Memory Ordering in C++ Implementation:

```cpp
// Writer perspective:
uint32_t r = read_idx.load(memory_order_acquire);     // 🔍 "See" all reader's previous writes
memcpy(&slots_[w], &data, sizeof(data));             // 📝 Write data
write_idx.store(nxt, memory_order_release);          // 🚀 "Publish" write completion

// Reader perspective:
uint32_t w = write_idx.load(memory_order_acquire);    // 🔍 "See" all writer's previous writes
// Now we're guaranteed to see the data that was written before write_idx was updated
memcpy(&data, &slots_[r], sizeof(data));             // 📖 Read data
read_idx.store(r+1, memory_order_release);           // 🚀 "Publish" read completion
```

#### Does This Prevent Simultaneous Read/Write?

**Short Answer**: It doesn't prevent simultaneous access to the **same slot**, but it prevents **data races** and **corruption**.

**Long Answer**: 
- ✅ **Prevents**: Reader reading partially written data
- ✅ **Prevents**: Writer overwriting data currently being read  
- ✅ **Prevents**: Index corruption from simultaneous updates
- ❌ **Doesn't prevent**: Reader and writer accessing different slots simultaneously (this is actually desired!)

### Race Condition Analysis

#### What CAN Go Wrong (Current Camera Implementation):
```python
# Thread 1 (Writer)                    # Thread 2 (Reader)
write_idx = get_write_idx()  # = 5     read_idx = get_read_idx()   # = 3
                                       # Reads slot 3 (good)
set_write_idx(6)             # RACE!   set_read_idx(4)             # RACE!
```

**Problem**: Non-atomic operations can be interleaved, causing:
- Lost updates
- ABA problems  
- Inconsistent state

#### What CANNOT Go Wrong (C++ Atomic Implementation):
```cpp
// Thread 1 (Writer)                          // Thread 2 (Reader)  
w = write_idx.load(memory_order_relaxed);     // Atomic read
r = read_idx.load(memory_order_acquire);      // Atomic read + sync barrier
// Data write happens here                    
write_idx.store(nxt, memory_order_release);   // Atomic write + sync barrier
```

**Protection**: Atomic operations are **indivisible** and memory ordering prevents reordering.






### Recommendation: Standardize on Atomic Ring Buffers

#### Proposed Unified Design:

```cpp
// Standard header for ALL shared memory ring buffers
struct AtomicRingBufferHeader {
    std::atomic<uint32_t> write_idx{0};
    std::atomic<uint32_t> read_idx{0}; 
    uint32_t capacity;               // Must be power of 2
    uint32_t slot_size;
    uint32_t element_type;           // Type identifier
    uint32_t reserved[3];            // Future use, alignment
};
```

#### Benefits of Standardization:

1. **Consistency**: Same semantics across all shared memory
2. **Performance**: Lock-free, wait-free for single producer/consumer
3. **Correctness**: Proper memory ordering prevents subtle bugs
4. **Portability**: Works across different architectures
5. **Scalability**: Multiple readers can work with atomic indices

#### Implementation Strategy:

1. **Phase 1**: Create atomic ring buffer template/base class
2. **Phase 2**: Migrate camera buffers to atomic implementation  
3. **Phase 3**: Migrate policy target to ring buffer (instead of linear array)
4. **Phase 4**: Standardize ACT-ReSiP on atomic ring buffers

#### Example Migration for Camera Buffers:

```python
# Instead of:
struct.pack_into("I", self.view, 0, write_idx)  # Non-atomic

# Use ctypes with proper alignment:
import ctypes
import mmap

class AtomicUInt32:
    def __init__(self, view, offset):
        self.view = view
        self.offset = offset
    
    def load(self):
        # Use atomic read on aligned boundary
        return struct.unpack_from("I", self.view, self.offset)[0]
    
    def store(self, value):
        # Use atomic write on aligned boundary  
        struct.pack_into("I", self.view, self.offset, value)
```

### Memory Ordering Requirements for Different Use Cases:

| Use Case | Writer Ordering | Reader Ordering | Justification |
|----------|----------------|----------------|---------------|
| Robot Control | `release` | `acquire` | Real-time, must see latest complete data |
| Camera Data | `release` | `acquire` | High bandwidth, need consistent frames |
| Policy Commands | `release` | `acquire` | Sequential consistency for command ordering |
| RL Flags | `seq_cst` | `seq_cst` | Complex multi-process synchronization |

**Key Insight**: Your suggestion about standardizing on atomic ring buffers is spot-on. The current mix of atomic (C++) and non-atomic (Python) implementations creates unnecessary complexity and potential race conditions.
