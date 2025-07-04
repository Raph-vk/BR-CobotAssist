// File: modules/robot/fanuc/cpp/control_loop.cpp

#include "control_loop.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <regex>
#include <algorithm>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/gil.h>
#include <ruckig/ruckig.hpp>

namespace py = pybind11;

//--------------------------------------------------------------------------------
// 32-bit unsigned integer arithmetic helpers for proper sequence ID handling
//--------------------------------------------------------------------------------
static uint32_t uint32_add(uint32_t a, uint32_t b) {
    return a + b;  // Unsigned overflow wraps naturally in C++
}

static uint32_t uint32_subtract(uint32_t a, uint32_t b) {
    return a - b;  // Unsigned underflow wraps naturally in C++
}

// Compare two 32-bit sequence IDs with proper overflow handling
// Returns true if 'a' is considered "less than" 'b' considering wraparound
static bool seq_id_less_than(uint32_t a, uint32_t b) {
    // Handle wraparound by checking the difference
    // If the difference is within half the range, use normal comparison
    uint32_t diff = b - a;
    return diff <= 0x7FFFFFFF;  // Half of 2^32
}

//--------------------------------------------------------------------------------
// Helpers for logging with time in the format: "2025-06-03 18:27:50,112"
//--------------------------------------------------------------------------------
static std::string currentTimeFormatted()
{
    using namespace std::chrono;
    auto now = system_clock::now();
    auto in_time_t = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    std::ostringstream ss;
    // Convert to broken-down time
    std::tm bt{};

#ifndef _WIN32
    localtime_r(&in_time_t, &bt);  // Linux
#endif
    ss << std::put_time(&bt, "%Y-%m-%d %H:%M:%S")
       << ',' << std::setw(3) << std::setfill('0') << ms.count();
    return ss.str();
}


//--------------------------------------------------------------------------------
// Ring-buffer writer implementation
//--------------------------------------------------------------------------------

ShmRingBufferWriter::ShmRingBufferWriter(int capacity, const std::string& shm_name, int dof) 
    : capacity_(capacity), shm_name_(shm_name) {
    
    bytes_ = sizeof(RingBufferHeader) + capacity_ * sizeof(JointDataSlot);

    fd_ = ::shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd_ < 0) throw std::runtime_error("shm_open failed");

    if (::ftruncate(fd_, static_cast<off_t>(bytes_)) != 0)
        throw std::runtime_error("ftruncate failed");

    void* base = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (base == MAP_FAILED) throw std::runtime_error("mmap failed");

    hdr_   = static_cast<RingBufferHeader*>(base);
    slots_ = reinterpret_cast<JointDataSlot*>(
                static_cast<char*>(base) + sizeof(RingBufferHeader));

    /* first creator initialises header */
    if (hdr_->slot_size != sizeof(JointDataSlot) || hdr_->capacity != static_cast<uint32_t>(capacity_)) {
        new (hdr_) RingBufferHeader{};
        hdr_->slot_size = sizeof(JointDataSlot);
        hdr_->capacity  = static_cast<uint32_t>(capacity_);
    }
    std::cout << currentTimeFormatted() << " [C++] ring-buffer ready\n";
}
ShmRingBufferWriter::~ShmRingBufferWriter() {
    if (hdr_) ::munmap(hdr_, bytes_);
    if (fd_ >= 0) ::close(fd_);
}
void ShmRingBufferWriter::push(const JointDataSlot& s, int dof) {
    uint32_t w = hdr_->write_idx.load(std::memory_order_relaxed);
    uint32_t r = hdr_->read_idx .load(std::memory_order_acquire);
    uint32_t nxt = (w + 1) & (static_cast<uint32_t>(capacity_) - 1);
    if (nxt == r) { // full => drop oldest
        hdr_->read_idx.store((r + 1) & (static_cast<uint32_t>(capacity_) - 1), std::memory_order_release);
    }
    // Copy the data, but only use the actual DOF values
    std::memcpy(&slots_[w], &s, sizeof(JointDataSlot));
    hdr_->write_idx.store(nxt, std::memory_order_release);
}


//--------------------------------------------------------------------------------
// Packet Structs
//--------------------------------------------------------------------------------
#pragma pack(push, 1)
struct RobotStatusPacketBE {
    uint32_t packet_type;
    uint32_t version_no;
    uint32_t seq_id_received;
    uint8_t  status;
    uint8_t  io_type;
    uint16_t io_index;
    uint16_t io_mask;
    uint16_t io_value;
    uint32_t timestamp;

    float x;
    float y;
    float z;
    float w;
    float p;
    float r;
    float ext1;
    float ext2;
    float ext3;
    float j1;
    float j2;
    float j3;
    float j4;
    float j5;
    float j6;
    float j7;
    float j8;
    float j9;
    float mc1;
    float mc2;
    float mc3;
    float mc4;
    float mc5;
    float mc6;
    float mc7;
    float mc8;
    float mc9;
};
#pragma pack(pop)
static constexpr size_t PACKET_SIZE_BYTES = sizeof(RobotStatusPacketBE);

#pragma pack(push, 1)
struct RobotCommandPacketBE {
    uint32_t packet_type;
    uint32_t version_no;
    uint32_t seq_id_sent;
    uint8_t  last_data;
    uint8_t  read_io_type;
    uint16_t read_io_index;
    uint16_t read_io_mask;
    uint8_t  data_format;
    uint8_t  writing_io_type;
    uint16_t writing_io_index;
    uint16_t writing_io_mask;
    uint16_t writing_io_value;
    uint8_t  pad[2];
    float    pos[9];
};
#pragma pack(pop)

//--------------------------------------------------------------------------------
// Time utilities
//--------------------------------------------------------------------------------
static double getEpochTimeSeconds() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto epoch_duration = now.time_since_epoch();
    return duration<double>(epoch_duration).count();
}

// Convert big-endian float => host float
static float ntohf(float be_float) {
    union {
        float f;
        uint8_t b[4];
    } src, dst;
    src.f = be_float;
    // big-endian => reverse
    dst.b[0] = src.b[3];
    dst.b[1] = src.b[2];
    dst.b[2] = src.b[1];
    dst.b[3] = src.b[0];
    return dst.f;
}

//--------------------------------------------------------------------------------
// Constructor
//--------------------------------------------------------------------------------
ControlLoop::ControlLoop(
    bool started_streaming,
    double control_dt,
    int dof,
    const std::vector<double>& start_position,
    const std::vector<std::array<double,3>>& joint_limits,
    bool joint_synchronization,
    int action_buffer_length,
    const std::vector<std::string>& server_address,
    const std::vector<std::string>& robot_address,
    int check_queue_period_divisor,
    bool play_recording_active,
    const std::deque<std::vector<double>>& master_positions,
    double gripper_treshold,
    double gripper_delay,
    double robot_speed,
    const std::vector<double>& target_pos_received,
    bool robot_running,
    bool recording,
    const std::vector<double>& upper_limits,
    const std::vector<double>& lower_limits,
    bool gripper_on,
    bool gripper_off,
    bool gripper_state,
    py::object py_callback,
    // Shared memory configuration parameters
    int shm_capacity,
    const std::string& shm_name,
    int shm_policy_capacity,
    const std::string& shm_policy_name,
    // Policy target shared memory parameters
    int shm_policy_target_capacity,
    const std::string& shm_policy_target_name,
    const std::string& shm_policy_target_format,
    int shm_policy_target_entry_size)
    
    : py_callback_(py_callback)  // IMPORTANT
{
    // Basic config
    started_streaming_    = started_streaming;
    control_dt_           = control_dt;
    dof_                  = dof;
    actual_dof_           = dof;  // Store actual DOF
    start_position_       = start_position;
    joint_limits_         = joint_limits;
    joint_synchronization_= joint_synchronization;
    action_buffer_length_ = action_buffer_length;

    // Parse server_address
    if (server_address.size() >= 2) {
        server_ip_   = server_address[0];
        server_port_ = static_cast<uint16_t>(std::stoi(server_address[1]));
    } else {
        std::cerr << currentTimeFormatted() << " [C++] Error: server_address must contain at least 2 elements (ip and port).\n";
    }

    // Parse robot_address
    if (robot_address.size() >= 2) {
        robot_ip_   = robot_address[0];
        robot_port_ = static_cast<uint16_t>(std::stoi(robot_address[1]));
    } else {
        std::cerr << currentTimeFormatted() << " [C++] Error: robot_address must contain at least 2 elements (ip and port).\n";
    }

    check_queue_period_divisor_ = check_queue_period_divisor;
    play_recording_active_      = play_recording_active;
    master_positions_           = master_positions;
    gripper_treshold_           = gripper_treshold;
    gripper_delay_              = gripper_delay;
    robot_speed_                = robot_speed;
    target_pos_received_        = target_pos_received;
    robot_running_              = robot_running;
    recording_                  = recording;
    upper_limits_               = upper_limits;
    lower_limits_               = lower_limits;
    gripper_on_                 = gripper_on;
    gripper_off_                = gripper_off;
    gripper_state_              = gripper_state;
    gripper_state_change_time_  = getEpochTimeSeconds();  // Initialize to current time

    seq_id_received_            = 0;
    seq_id_sent_                = 0;
    started_receiving_motion_stream_ = false;
    udp_running_                = false;
    joint_state_received_.resize(dof_, 0.0);

    // Create UDP socket
    udp_socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    std::cout << currentTimeFormatted() 
              << " [C++] Creating UDP socket, fd=" << udp_socket_fd_ << "\n";
    if (udp_socket_fd_ < 0) {
        std::cerr << currentTimeFormatted() << " [C++] Error creating socket.\n";
    }

    // Bind
    sockaddr_in local_addr;
    std::memset(&local_addr, 0, sizeof(local_addr));
    local_addr.sin_family      = AF_INET;
    local_addr.sin_addr.s_addr = inet_addr(server_ip_.c_str()); 
    local_addr.sin_port        = htons(server_port_);

    if (bind(udp_socket_fd_, reinterpret_cast<sockaddr*>(&local_addr), sizeof(local_addr)) < 0) {
        std::cerr << currentTimeFormatted()
                  << " [C++] Error binding UDP socket to "
                  << server_ip_ << ":" << server_port_ << "\n";
    } else {
        std::cout << currentTimeFormatted()
                  << " [C++] UDP socket bound to "
                  << server_ip_ << ":" << server_port_ << "\n";
    }

    // Store shared memory configuration
    shm_capacity_ = shm_capacity;
    shm_name_ = shm_name;
    shm_policy_capacity_ = shm_policy_capacity;
    shm_policy_name_ = shm_policy_name;
    shm_policy_target_capacity_ = shm_policy_target_capacity;
    shm_policy_target_name_ = shm_policy_target_name;
    shm_policy_target_format_ = shm_policy_target_format;
    shm_policy_target_entry_size_ = shm_policy_target_entry_size;

    // Create ring-buffer writers with configurable parameters
    try { 
        shm_writer_ = std::make_unique<ShmRingBufferWriter>(shm_capacity_, shm_name_, actual_dof_); 
        shm_policy_writer_ = std::make_unique<ShmRingBufferWriter>(shm_policy_capacity_, shm_policy_name_, actual_dof_);
    }
    catch (const std::exception& e) {
        std::cerr << currentTimeFormatted() << " [C++] shm setup failed: "
                  << e.what() << '\n';
    }
    
    // Initialize policy target shared memory reader
    try {
        std::cout << currentTimeFormatted() << " [C++] Opening policy target shared memory: " 
                  << shm_policy_target_name_ << std::endl;
        shm_policy_target_fd_ = ::shm_open(shm_policy_target_name_.c_str(), O_RDONLY, 0);
        if (shm_policy_target_fd_ == -1) {
            std::cerr << currentTimeFormatted() << " [C++] Failed to open policy target shared memory" << std::endl;
        } else {
            shm_policy_target_size_ = shm_policy_target_capacity_ * shm_policy_target_entry_size_;
            shm_policy_target_ptr_ = ::mmap(nullptr, shm_policy_target_size_, PROT_READ, MAP_SHARED, shm_policy_target_fd_, 0);
            if (shm_policy_target_ptr_ == MAP_FAILED) {
                std::cerr << currentTimeFormatted() << " [C++] Failed to map policy target shared memory" << std::endl;
                shm_policy_target_ptr_ = nullptr;
            } else {
                std::cout << currentTimeFormatted() << " [C++] Policy target shared memory mapped successfully" << std::endl;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << currentTimeFormatted() << " [C++] policy target shm setup failed: "
                  << e.what() << '\n';
    }
}

ControlLoop::~ControlLoop()
{
    stopControlLoop();
    
    // Cleanup policy target shared memory
    if (shm_policy_target_ptr_ && shm_policy_target_ptr_ != MAP_FAILED) {
        ::munmap(shm_policy_target_ptr_, shm_policy_target_size_);
        shm_policy_target_ptr_ = nullptr;
    }
    if (shm_policy_target_fd_ != -1) {
        ::close(shm_policy_target_fd_);
        shm_policy_target_fd_ = -1;
    }
}


//--------------------------------------------------------------------------------
// Publish sample to shared memory
//--------------------------------------------------------------------------------
/*
 * This function ensures the C++ version sends exactly the same data as the Python version:
 * Python:                           C++:
 * "master_position"          ->     master_pos (from inp.target_position)
 * "send_position_robot"      ->     send_position_robot (from out.new_position)  
 * "robot_position"           ->     robot_pos (from joint_state_received_)
 * "robot_position_timestamp" ->     robot_position_timestamp (from joint_state_received_time_)
 * "gripper_on"               ->     gripper_on
 * "gripper_off"              ->     gripper_off  
 * "seq_id"                   ->     seq_id
 * "timestamp"                ->     timestamp (current time when data is written)
 */

void ControlLoop::publishSampleToShm(const std::vector<double>& master_position,
                                 const std::vector<double>& send_position_robot,
                                 const std::vector<double>& robot_position,
                                 double robot_position_timestamp,
                                 bool g_on, bool g_off)
{
    if (!shm_writer_) return;
    JointDataSlot s{};  // Zero-initialize the entire structure
    
    // Copy position arrays with gripper state included as last element
    // Each array should have actual_dof_ + 1 elements (6 joints + 1 gripper)
    int total_dof = actual_dof_ + 1;  // 6 + 1 = 7
    
    if (master_position.size() >= total_dof) {
        std::memcpy(s.master_pos, master_position.data(), total_dof*sizeof(double));
    } else if (master_position.size() >= actual_dof_) {
        // Fallback: copy joints and add gripper state
        std::memcpy(s.master_pos, master_position.data(), actual_dof_*sizeof(double));
        s.master_pos[actual_dof_] = g_on ? 1.0 : 0.0;  // Add gripper state as last element
    }
    
    if (send_position_robot.size() >= total_dof) {
        std::memcpy(s.send_position_robot, send_position_robot.data(), total_dof*sizeof(double));
    } else if (send_position_robot.size() >= actual_dof_) {
        // Fallback: copy joints and add gripper state
        std::memcpy(s.send_position_robot, send_position_robot.data(), actual_dof_*sizeof(double));
        s.send_position_robot[actual_dof_] = g_on ? 1.0 : 0.0;  // Add gripper state as last element
    }
    
    if (robot_position.size() >= total_dof) {
        std::memcpy(s.robot_pos, robot_position.data(), total_dof*sizeof(double));
    } else if (robot_position.size() >= actual_dof_) {
        // Fallback: copy joints and add gripper state  
        std::memcpy(s.robot_pos, robot_position.data(), actual_dof_*sizeof(double));
        s.robot_pos[actual_dof_] = g_on ? 1.0 : 0.0;  // Add gripper state as last element
    }
    
    // Keep gripper flags for backward compatibility with Python reader
    s.gripper_on  = g_on;
    s.gripper_off = g_off;
    s.seq_id      = seq_id_sent_;
    s.timestamp   = nowSec();
    s.robot_position_timestamp = robot_position_timestamp;
    shm_writer_->push(s, total_dof);  // Pass total DOF including gripper
}

//--------------------------------------------------------------------------------
// Publish sample to policy shared memory
//--------------------------------------------------------------------------------
void ControlLoop::publishSampleToPolicyShm(const std::vector<double>& master_position,
                                       const std::vector<double>& send_position_robot,
                                       const std::vector<double>& robot_position,
                                       double robot_position_timestamp,
                                       bool g_on, bool g_off)
{
    if (!shm_policy_writer_) return;
    JointDataSlot s{};  // Zero-initialize the entire structure
    
    // Copy position arrays with gripper state included as last element
    // Each array should have actual_dof_ + 1 elements (6 joints + 1 gripper)
    int total_dof = actual_dof_ + 1;  // 6 + 1 = 7
    
    if (master_position.size() >= total_dof) {
        std::memcpy(s.master_pos, master_position.data(), total_dof*sizeof(double));
    } else if (master_position.size() >= actual_dof_) {
        // Fallback: copy joints and add gripper state
        std::memcpy(s.master_pos, master_position.data(), actual_dof_*sizeof(double));
        s.master_pos[actual_dof_] = g_on ? 1.0 : 0.0;  // Add gripper state as last element
    }
    
    if (send_position_robot.size() >= total_dof) {
        std::memcpy(s.send_position_robot, send_position_robot.data(), total_dof*sizeof(double));
    } else if (send_position_robot.size() >= actual_dof_) {
        // Fallback: copy joints and add gripper state
        std::memcpy(s.send_position_robot, send_position_robot.data(), actual_dof_*sizeof(double));
        s.send_position_robot[actual_dof_] = g_on ? 1.0 : 0.0;  // Add gripper state as last element
    }
    
    if (robot_position.size() >= total_dof) {
        std::memcpy(s.robot_pos, robot_position.data(), total_dof*sizeof(double));
    } else if (robot_position.size() >= actual_dof_) {
        // Fallback: copy joints and add gripper state  
        std::memcpy(s.robot_pos, robot_position.data(), actual_dof_*sizeof(double));
        s.robot_pos[actual_dof_] = g_on ? 1.0 : 0.0;  // Add gripper state as last element
    }
    
    // Keep gripper flags for backward compatibility with Python reader
    s.gripper_on  = g_on;
    s.gripper_off = g_off;
    s.seq_id      = seq_id_sent_;
    s.timestamp   = nowSec();
    s.robot_position_timestamp = robot_position_timestamp;
    shm_policy_writer_->push(s, total_dof);  // Pass total DOF including gripper
}

//--------------------------------------------------------------------------------
// Public API
//--------------------------------------------------------------------------------
void ControlLoop::startControlLoop(bool started_streaming)
{
    started_streaming_ = started_streaming;
    control_thread_ = std::thread(&ControlLoop::controlLoop, this, started_streaming);
}

void ControlLoop::stopControlLoop()
{
    robot_running_ = false;

    if (control_thread_.joinable()) {
        control_thread_.join();
        std::cout << currentTimeFormatted() 
                  << " [C++] Control loop stopped and joined\n";
    }

    stopReceiving();

    if (udp_socket_fd_ >= 0) {
        ::close(udp_socket_fd_);
        udp_socket_fd_ = -1;
    }
}

void ControlLoop::startReceiving()
{
    udp_running_ = true;
    udp_thread_  = std::thread(&ControlLoop::receiveLoop, this);
}

void ControlLoop::stopReceiving()
{
    if (udp_running_) {
        udp_running_ = false;
        if (udp_thread_.joinable()) {
            udp_thread_.join();
        }
    }
}

void ControlLoop::setRecording(bool rec)
{
    recording_ = rec;
}

//--------------------------------------------------------------------------------
// Internal
//--------------------------------------------------------------------------------
void ControlLoop::controlLoop(bool started_streaming)
{
    std::cout << currentTimeFormatted() << " [C++] control_loop, starting receive loop\n";
    startReceiving();
    std::cout << currentTimeFormatted() << " [C++] control_loop, receive loop started\n";

    // Setup Ruckig
    std::cout << currentTimeFormatted() << " [C++] control_loop, Setup Ruckig\n";
    setupRuckig();
    std::cout << currentTimeFormatted() << " [C++] control_loop, Ruckig setup done\n";

    // Wait for first status
    while (!started_receiving_motion_stream_ && robot_running_) {
        std::this_thread::sleep_for(std::chrono::duration<double>(control_dt_ / 4.0));
        std::cout << currentTimeFormatted() 
                  << " [C++] control_loop, waiting for first status...\n";
    }

    std::cout << currentTimeFormatted() 
              << " [C++] control_loop, first status received\n";

    if (!robot_running_) {
        return;
    }

    // wait for 0.1 seconds
    std::this_thread::sleep_for(std::chrono::duration<double>(0.1));

    // Fill action buffer
    std::cout << currentTimeFormatted() 
              << " [C++] control_loop, streaming started, filling buffer\n";
    for (int i = 0; i < action_buffer_length_; i++) {
        sendJointPos(start_position_, gripper_on_, gripper_off_);
    }

    std::cout << currentTimeFormatted()
              << " [C++] control_loop, starting control loop\n";
    std::vector<double> previous_action_master = start_position_;

    // Main loop
    while (robot_running_) {
        while (seq_id_less_than(uint32_add(seq_id_received_, action_buffer_length_), seq_id_sent_)) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(control_dt_ / (check_queue_period_divisor_ / 4.0)));
        }

        // If playing a recorded sequence
        std::vector<double> action_master;
        if (play_recording_active_) {
            if (!master_positions_.empty()) {
                action_master = master_positions_.front();
                master_positions_.pop_front();
            } else {
                std::cout << currentTimeFormatted() 
                          << " [C++] control_loop, recording playback completed\n";
                break;
            }
        } else if (run_policy_active_) {
            // Policy is active, get next action from shared memory
            action_master = getNextPolicyAction();
            if (action_master.empty()) {
                // If no policy action available, use previous action or default
                if (!target_pos_received_.empty()) {
                    action_master = target_pos_received_;
                } else {
                    // Use a safe default or break
                    std::cout << currentTimeFormatted() 
                              << " [C++] No policy action available, stopping\n";
                    break;
                }
            }
        } else {
            // Otherwise, use target_pos_received_
            if (!target_pos_received_.empty()) {
                action_master = target_pos_received_;
            } else {
                break;
            }
        }

        // Ruckig step
        auto current_position = updateRuckigInput(action_master, *inp_, previous_action_master);
        previous_action_master = action_master;

        bool success_calc = trajectoryCalculation(*otg_, *inp_, *out_, current_position);
        determineGripperState(action_master.back());
        if (!success_calc) {
            std::cerr << currentTimeFormatted()
                      << " [C++] control_loop, trajectory calculation failed\n";
            break;
        }

        std::vector<double> send_position_robot  = out_->new_position;
        std::vector<double> master_position = inp_->target_position;

        bool success_send = sendJointPos(send_position_robot, gripper_on_, gripper_off_);
        
        // Store current gripper flags before resetting them for shared memory
        bool gripper_on_for_shm = gripper_on_;
        bool gripper_off_for_shm = gripper_off_;
        
        if (gripper_on_)  { gripper_on_  = false; }
        if (gripper_off_) { gripper_off_ = false; }

        if (!success_send) {
            std::cerr << currentTimeFormatted()
                      << " [C++] control_loop, could not send joint position\n";
            break;
        }

        // If recording => store to save interface shared memory (use captured gripper flags)
        if (recording_) {
            publishSampleToShm(master_position, send_position_robot, joint_state_received_, 
                              joint_state_received_time_, gripper_on_for_shm, gripper_off_for_shm);
        }
        
        // Always publish to policy interface shared memory for real-time control
        publishSampleToPolicyShm(master_position, send_position_robot, joint_state_received_, 
                                joint_state_received_time_, gripper_on_for_shm, gripper_off_for_shm);

    } // end main control loop

    // If streaming => stop robot
    if (started_streaming_) {
        std::cout << currentTimeFormatted() 
                  << " [C++] control_loop, stopping robot\n";
        stopRobot(*otg_, *inp_, *out_);
        std::cout << currentTimeFormatted() 
                  << " [C++] control_loop, robot stopped\n";
    }

    {
        // aquire the python GIL
        std::cout << currentTimeFormatted() 
                  << " [C++] control_loop, calling Python callback if set\n";
        pybind11::gil_scoped_acquire gil;
        std::cout << currentTimeFormatted() 
                  << " [C++] control_loop, Python callback acquired\n";
        if (!py_callback_.is_none()) {
            std::cout << currentTimeFormatted() 
                      << " [C++] control_loop, calling Python callback because not none\n";
            try {
                py_callback_();
                std::cout << currentTimeFormatted() 
                          << " [C++] control_loop, Python callback called successfully\n";
            } catch (const std::exception& e) {
                std::cerr << currentTimeFormatted()
                          << " [C++] control_loop, error in Python callback: "
                          << e.what() << std::endl;
            }
        }
        std::cout << currentTimeFormatted() 
                  << " [C++] control_loop, Python callback done\n";    
    }

    // stop receiving
    std::cout << currentTimeFormatted()
              << " [C++] control_loop, stopping receive loop\n";
    stopReceiving();
    std::cout << currentTimeFormatted()
              << " [C++] control_loop, receive loop stopped\n";
}

void ControlLoop::receiveLoop()
{
    std::cout << currentTimeFormatted() 
              << " [C++] receiveLoop, starting\n";

    while (udp_running_) {
        uint8_t buffer[4096];
        std::memset(buffer, 0, sizeof(buffer));
        sockaddr_in src_addr;
        socklen_t addr_len = sizeof(src_addr);

        int bytes_received = ::recvfrom(
            udp_socket_fd_,
            buffer,
            sizeof(buffer),
            0,
            reinterpret_cast<sockaddr*>(&src_addr),
            &addr_len
        );

        if (!udp_running_) break;

        if (bytes_received < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            } else {
                continue;
            }
        }
        if (bytes_received == 0) {
            continue;
        }

        started_receiving_motion_stream_ = true;
        decodeStatusPacket(buffer, static_cast<size_t>(bytes_received));
    }

    std::cout << currentTimeFormatted()
              << " [C++] receiveLoop, exiting\n";
}

void ControlLoop::decodeStatusPacket(const uint8_t* data, size_t size)
{
    if (size < PACKET_SIZE_BYTES) {
        std::cerr << currentTimeFormatted()
                  << " [C++] decodeStatusPacket: Packet too small.\n";
        return;
    }

    RobotStatusPacketBE pktBE;
    std::memcpy(&pktBE, data, PACKET_SIZE_BYTES);

    uint32_t seq_id_be = ntohl(pktBE.seq_id_received);
    uint16_t io_val_be = ntohs(pktBE.io_value);

    float j1 = ntohf(pktBE.j1);
    float j2 = ntohf(pktBE.j2);
    float j3 = ntohf(pktBE.j3);
    float j4 = ntohf(pktBE.j4);
    float j5 = ntohf(pktBE.j5);
    float j6 = ntohf(pktBE.j6);

    seq_id_received_ = seq_id_be;
    // std::cout << currentTimeFormatted() 
    //           << " [C++] decodeStatusPacket, seq_id_received=" << seq_id_received_ << "\n";

    uint16_t io_val = io_val_be;
    if (io_val == 2) {
        io_val = 1;
    }

    std::vector<double> new_joint_state {
        static_cast<double>(j1),
        static_cast<double>(j2),
        static_cast<double>(j3),
        static_cast<double>(j4),
        static_cast<double>(j5),
        static_cast<double>(j6),
        static_cast<double>(io_val)
    };
    new_joint_state = J3InteractionRev(new_joint_state);

    joint_state_received_      = new_joint_state;
    joint_state_received_time_ = getEpochTimeSeconds();
}

//--------------------------------------------------------------------------------
// sendJointPos
//--------------------------------------------------------------------------------
bool ControlLoop::sendJointPos(const std::vector<double>& in_position,
                           bool gripper_on, bool gripper_off)
{
    if (seq_id_sent_ == 0) {
        seq_id_sent_ = seq_id_received_ - 1;
    }
    seq_id_sent_++;

    uint32_t packet_type = 1;
    uint32_t version_no  = 1;
    uint8_t  last_data   = 0;
    uint8_t  read_io_type  = 9;
    uint16_t read_io_index = 1;
    uint16_t read_io_mask  = 0x0002;
    uint8_t  data_format   = 1;

    uint8_t  writing_io_type  = 0;
    uint16_t writing_io_index = 0;
    uint16_t writing_io_mask  = 0;
    uint16_t writing_io_value = 0;

    if (gripper_on) {
        writing_io_type  = 9;
        writing_io_index = 1;
        writing_io_mask  = 0x0002;
        writing_io_value = 0x0002;
    } else if (gripper_off) {
        writing_io_type  = 9;
        writing_io_index = 1;
        writing_io_mask  = 0x0002;
        writing_io_value = 0x0000;
    }

    std::vector<double> position = in_position;
    position.resize(9, 0.0);
    position = J3Interaction(position);  // J3 offset

    RobotCommandPacketBE cmd;
    std::memset(&cmd, 0, sizeof(cmd));

    cmd.packet_type   = htonl(packet_type);
    cmd.version_no    = htonl(version_no);
    cmd.seq_id_sent   = htonl(seq_id_sent_);
    cmd.last_data     = last_data;
    cmd.read_io_type  = read_io_type;
    cmd.read_io_index = htons(read_io_index);
    cmd.read_io_mask  = htons(read_io_mask);
    cmd.data_format   = data_format;

    cmd.writing_io_type   = writing_io_type;
    cmd.writing_io_index  = htons(writing_io_index);
    cmd.writing_io_mask   = htons(writing_io_mask);
    cmd.writing_io_value  = htons(writing_io_value);

    for (int i = 0; i < 9; i++) {
        float fval = static_cast<float>(position[i]);
        union { float f; uint8_t b[4]; } src, dst;
        src.f = fval;
        dst.b[0] = src.b[3];
        dst.b[1] = src.b[2];
        dst.b[2] = src.b[1];
        dst.b[3] = src.b[0];
        cmd.pos[i] = dst.f;
    }

    // Destination
    sockaddr_in remote_addr;
    std::memset(&remote_addr, 0, sizeof(remote_addr));
    remote_addr.sin_family      = AF_INET;
    remote_addr.sin_port        = htons(robot_port_);
    remote_addr.sin_addr.s_addr = inet_addr(robot_ip_.c_str());

    const uint8_t* raw_bytes = reinterpret_cast<const uint8_t*>(&cmd);
    size_t packet_size = sizeof(cmd);

    int rc = ::sendto(
        udp_socket_fd_,
        raw_bytes,
        packet_size,
        0,
        reinterpret_cast<const sockaddr*>(&remote_addr),
        sizeof(remote_addr)
    );
    if (rc < 0) {
        // std::cerr << currentTimeFormatted() 
        //           << " [C++] sendJointPos, error sending packet: errno=" << errno << std::endl;
        return false;
    }

    double now = getEpochTimeSeconds();
    double dt  = now - time_last_sent_;
    if (dt > 0.008 && dt < 1e9) {
        std::cerr << currentTimeFormatted() 
                  << " [C++] sendJointPos, warning: last packet dt=" << dt
                  << ", packet=" << seq_id_sent_ << std::endl;
    } else {
        std::cout << currentTimeFormatted()
                  << " [C++] sendJointPos, sent packet " << seq_id_sent_ 
                  << " successfully\n";
    }
    time_last_sent_ = now;
    return true;
}

//--------------------------------------------------------------------------------
// setupRuckig
//--------------------------------------------------------------------------------
void ControlLoop::setupRuckig()
{
    otg_ = std::make_unique<ruckig::Ruckig<0>>(dof_, control_dt_);
    inp_ = std::make_unique<ruckig::InputParameter<0>>(dof_);
    out_ = std::make_unique<ruckig::OutputParameter<0>>(dof_);

    inp_->current_position     = start_position_;
    inp_->current_velocity     = std::vector<double>(dof_, 0.0);
    inp_->current_acceleration = std::vector<double>(dof_, 0.0);

    inp_->target_position     = start_position_;
    inp_->target_velocity     = std::vector<double>(dof_, 0.0);
    inp_->target_acceleration = std::vector<double>(dof_, 0.0);

    // Compute velocity/acc/jerk
    std::vector<double> max_velocity(dof_);
    std::vector<double> max_acceleration(dof_);
    std::vector<double> max_jerk(dof_);

    for (int i = 0; i < dof_; i++) {
        double v = joint_limits_[i][0] * robot_speed_;
        double a = joint_limits_[i][1] * (robot_speed_ * robot_speed_);
        double j = joint_limits_[i][2] * (robot_speed_ * robot_speed_ * robot_speed_);

        v = std::round(v*1e6)/1e6;
        a = std::round(a*1e6)/1e6;
        j = std::round(j*1e6)/1e6;

        max_velocity[i]     = v;
        max_acceleration[i] = a;
        max_jerk[i]         = j;
    }

    inp_->max_velocity     = max_velocity;
    inp_->max_acceleration = max_acceleration;
    inp_->max_jerk         = max_jerk;

    if (!joint_synchronization_) {
        inp_->synchronization = ruckig::Synchronization::None;
    }
}

//--------------------------------------------------------------------------------
// updateRuckigInput
//--------------------------------------------------------------------------------
std::vector<double> ControlLoop::updateRuckigInput(
    const std::vector<double>& action_master_in,
    ruckig::InputParameter<0>& inp,
    const std::vector<double>& previous_action_master)
{
    std::vector<double> current_position = roundPosition(inp.current_position);
    inp.current_position = current_position;

    std::vector<double> action_master = action_master_in;

    // clamp
    for (int i = 0; i < dof_; i++) {
        if (action_master[i] > upper_limits_[i]) {
            action_master[i] = upper_limits_[i];
        } else if (action_master[i] < lower_limits_[i]) {
            action_master[i] = lower_limits_[i];
        }
    }
    // // "stick" if difference < threshold
    // for (int i = 0; i < dof_; i++) {
    //     double diff_curr = std::fabs(action_master[i] - current_position[i]);
    //     double diff_prev = std::fabs(action_master[i] - previous_action_master[i]);
    //     if (diff_curr < 0.2 && diff_prev < 0.2) {
    //         action_master[i] = previous_action_master[i];
    //     }
    // }

    inp.target_position = action_master;
    return current_position;
}

//--------------------------------------------------------------------------------
// trajectoryCalculation
//--------------------------------------------------------------------------------
bool ControlLoop::trajectoryCalculation(
    ruckig::Ruckig<0>& otg,
    ruckig::InputParameter<0>& inp,
    ruckig::OutputParameter<0>& out,
    const std::vector<double>& /* current_position */)
{
    double start_time = getEpochTimeSeconds();

    try {
        auto result = otg.update(inp, out);
        // we do not check result here explicitly
    }
    catch (const std::exception& e) {
        bool success = handleTrajectoryError(e, inp, out, otg, dof_);
        if (!success) {
            std::cerr << currentTimeFormatted()
                      << " [C++] trajectoryCalculation: Ruckig error => "
                      << e.what() << std::endl;
            return false;
        } else {
            out.pass_to_input(inp);
        }
    }
    out.pass_to_input(inp);

    double elapsed = getEpochTimeSeconds() - start_time;
    if (elapsed > control_dt_) {
        std::cerr << currentTimeFormatted()
                  << " [C++] trajectoryCalculation took " << elapsed
                  << " s > " << control_dt_ << " s\n";
    }
    return true;
}

//--------------------------------------------------------------------------------
// handleTrajectoryError
//--------------------------------------------------------------------------------
bool ControlLoop::handleTrajectoryError(
    const std::exception& e,
    ruckig::InputParameter<0>& inp,
    ruckig::OutputParameter<0>& out,
    ruckig::Ruckig<0>& otg,
    int dof)
{
    std::string msg = e.what();
    static const std::regex rgx("dof:\\s*(\\d+)");
    std::smatch match;
    if (!std::regex_search(msg, match, rgx)) {
        return false;
    }

    int error_dof = std::stoi(match[1].str());
    try {
        // Nudging the target
        auto target = inp.target_position;
        auto vel    = inp.current_velocity;
        auto acc    = inp.current_acceleration;
        for (int i = 0; i < dof; i++) {
            vel[i] = std::round(vel[i]*1e6)/1e6;
            acc[i] = std::round(acc[i]*1e6)/1e6;
        }
        target[error_dof] += 0.0001;
        inp.target_position = target;
        inp.current_velocity = vel;
        inp.current_acceleration = acc;

        otg.update(inp, out);
        out.pass_to_input(inp);
        return true;
    }
    catch (const std::exception& e2) {
        // second attempt
        std::string msg2 = e2.what();
        std::smatch match2;
        if (!std::regex_search(msg2, match2, rgx)) {
            return false;
        }
        int error2_dof = std::stoi(match2[1].str());
        try {
            auto target   = inp.target_position;
            auto curr_pos = inp.current_position;
            target[error2_dof] = curr_pos[error2_dof];
            inp.target_position = target;

            otg.update(inp, out);
            std::cerr << currentTimeFormatted()
                      << " [C++] Ruckig cannot calculate. Sent DOF "
                      << error2_dof << " to current position.\n";
            out.pass_to_input(inp);
            return true;
        }
        catch (const std::exception& e3) {
            std::cerr << currentTimeFormatted()
                      << " [C++] Ruckig cannot calculate: "
                      << e3.what() << std::endl;
            error_3_occur_++;
            if (error_3_occur_ > 5) {
                return false;
            }
            // fallback
            auto curr_pos  = inp.current_position;
            auto curr_vel  = inp.current_velocity;
            auto curr_acc  = inp.current_acceleration;
            auto max_vel   = inp.max_velocity;
            auto max_acc   = inp.max_acceleration;
            auto max_jerk  = inp.max_jerk;

            for (int i=0; i<dof; i++) {
                double sign_vel    = (curr_vel[i]>0.0) ? 1.0 : (curr_vel[i]<0.0)? -1.0 : 0.0;
                double desired_acc = -sign_vel * max_acc[i];
                double jerk_limit  = max_jerk[i] * control_dt_ * 0.8;
                double delta       = std::min(std::max(desired_acc - curr_acc[i], -jerk_limit), jerk_limit);

                double next_acc = std::min(std::max(curr_acc[i] + delta, -max_acc[i]), max_acc[i]);
                double next_vel = std::min(std::max(curr_vel[i] + next_acc*control_dt_, -max_vel[i]), max_vel[i]);
                double next_pos = curr_pos[i]
                                  + curr_vel[i]*control_dt_
                                  + 0.5*curr_acc[i]*(control_dt_*control_dt_);

                curr_pos[i] = next_pos;
                curr_vel[i] = next_vel;
                curr_acc[i] = next_acc;
            }
            inp.current_position     = curr_pos;
            inp.current_velocity     = curr_vel;
            inp.current_acceleration = curr_acc;
            return true;
        }
    }
    return false;
}

//--------------------------------------------------------------------------------
// stopRobot
//--------------------------------------------------------------------------------
bool ControlLoop::stopRobot(ruckig::Ruckig<0>& otg,
                        ruckig::InputParameter<0>& inp,
                        ruckig::OutputParameter<0>& out)
{
    std::cout << currentTimeFormatted()
              << " [C++] stopRobot, reducing speed\n";

    ruckig::Result res = ruckig::Result::Working;
    int nr_stop_actions = 50;
    int stop_step_counter = 0;

    while (res == ruckig::Result::Working) {
        while (seq_id_less_than(uint32_add(seq_id_received_, action_buffer_length_), seq_id_sent_)) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(control_dt_ / check_queue_period_divisor_));
        }
        res = otg.update(*inp_, *out_);
        out_->pass_to_input(*inp_);
        auto puppet = out_->new_position;
        sendJointPos(puppet, gripper_on_, gripper_off_);
    }

    std::cout << currentTimeFormatted()
              << " [C++] stopRobot, stopped robot\n";
    std::cout << currentTimeFormatted()
              << " [C++] stopRobot, sending " << nr_stop_actions << " more actions\n";

    while (stop_step_counter < nr_stop_actions) {
        while (seq_id_less_than(uint32_add(seq_id_received_, action_buffer_length_), seq_id_sent_)) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(control_dt_ / check_queue_period_divisor_));
        }
        if (gripper_on_) {
            gripper_off_ = true;
        }
        stop_step_counter++;
        auto puppet = out_->new_position;
        sendJointPos(puppet, gripper_on_, gripper_off_);
    }

    std::cout << currentTimeFormatted()
              << " [C++] stopRobot, done stopping robot\n";
    return true;
}

//--------------------------------------------------------------------------------
// determineGripperState
//--------------------------------------------------------------------------------
void ControlLoop::determineGripperState(double gripper_state_val)
{
    double now = getEpochTimeSeconds();
    gripper_on_ = false;
    gripper_off_ = false;

    // Turn ON
    if (gripper_state_val >= gripper_treshold_ && !gripper_state_) {
        std::cout << currentTimeFormatted() << " [C++] Gripper turning ON: value=" 
                  << gripper_state_val << " >= threshold=" << gripper_treshold_ << "\n";
        if (gripper_delay_ > 0.0) {
            gripper_state_change_time_threshold_ = gripper_delay_ / robot_speed_;
            // Check if enough time HAS passed since last state change
            if ((now - gripper_state_change_time_) >= gripper_state_change_time_threshold_) {
                gripper_on_ = true;
                gripper_state_ = true;
                double elapsed_time = now - gripper_state_change_time_;
                gripper_state_change_time_ = now;
                std::cout << currentTimeFormatted() << " [C++] Gripper ON activated after delay of " 
                          << elapsed_time << "s\n";
            } else {
                std::cout << currentTimeFormatted() << " [C++] Gripper ON delayed: " 
                          << (now - gripper_state_change_time_) << "s < " 
                          << gripper_state_change_time_threshold_ << "s required\n";
            }
        } else {
            gripper_on_ = true;
            gripper_state_ = true;
            gripper_state_change_time_ = now;
            std::cout << currentTimeFormatted() << " [C++] Gripper ON activated immediately (no delay)\n";
        }
    } 
    // Turn OFF
    else if (gripper_state_val < gripper_treshold_ && gripper_state_) {
        std::cout << currentTimeFormatted() << " [C++] Gripper turning OFF: value=" 
                  << gripper_state_val << " < threshold=" << gripper_treshold_ << "\n";
        gripper_off_ = true;
        gripper_state_ = false;
        gripper_state_change_time_ = now;
    }
}

//--------------------------------------------------------------------------------
// roundPosition
//--------------------------------------------------------------------------------
std::vector<double> ControlLoop::roundPosition(const std::vector<double>& position)
{
    std::vector<double> rounded = position;
    for (auto &val : rounded) {
        val = std::round(val * 1e6) / 1e6;
    }
    return rounded;
}

//--------------------------------------------------------------------------------
// J3Interaction
//--------------------------------------------------------------------------------
std::vector<double> ControlLoop::J3Interaction(const std::vector<double>& action)
{
    auto result = action;
    if (result.size() > 2) {
        // result[2] = result[2] - result[1]
        result[2] = result[2] - result[1];
    }
    return result;
}

//--------------------------------------------------------------------------------
// J3InteractionRev
//--------------------------------------------------------------------------------
std::vector<double> ControlLoop::J3InteractionRev(const std::vector<double>& action)
{
    auto result = action;
    if (result.size() > 2) {
        // result[2] = result[2] + result[1]
        result[2] = result[2] + result[1];
    }
    return result;
}

double ControlLoop::nowSec() const
{
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

//--------------------------------------------------------------------------------
// Get next policy action from shared memory
//--------------------------------------------------------------------------------
std::vector<double> ControlLoop::getNextPolicyAction()
{
    if (!shm_policy_target_ptr_ || !run_policy_active_) {
        return {};
    }
    
    try {
        // Calculate the next sequence ID we need
        uint32_t next_seq_id = seq_id_sent_ + 1;
        
        // Helper to read and unpack data at a specific buffer index
        auto read_buffer_entry = [this](int buffer_index) -> std::pair<uint32_t, std::vector<double>> {
            size_t offset = buffer_index * shm_policy_target_entry_size_;
            const uint8_t* data = static_cast<const uint8_t*>(shm_policy_target_ptr_) + offset;
            
            // Unpack sequence ID (first 4 bytes as uint32_t)
            uint32_t seq_id;
            std::memcpy(&seq_id, data, sizeof(uint32_t));
            
            // Unpack joint positions (remaining bytes as doubles)
            std::vector<double> positions;
            const double* pos_data = reinterpret_cast<const double*>(data + sizeof(uint32_t));
            int num_joints = (shm_policy_target_entry_size_ - sizeof(uint32_t)) / sizeof(double);
            positions.assign(pos_data, pos_data + num_joints);
            
            return {seq_id, positions};
        };
        
        // First, try the expected modulo position (fast path)
        int expected_index = next_seq_id % shm_policy_target_capacity_;
        auto [seq_id, joint_positions] = read_buffer_entry(expected_index);
        
        if (seq_id == next_seq_id) {
            // std::cout << currentTimeFormatted() 
            //           << " [C++] Found policy action for seq_id " << seq_id 
            //           << " at expected index " << expected_index << std::endl;
            return joint_positions;
        }
        
        // If not found at expected position, search the buffer
        std::cout << currentTimeFormatted() 
                  << " [C++] seq_id " << next_seq_id 
                  << " not at expected index " << expected_index 
                  << " (found " << seq_id << "), searching buffer..." << std::endl;
        
        for (int buffer_index = 0; buffer_index < shm_policy_target_capacity_; ++buffer_index) {
            if (buffer_index == expected_index) {
                continue;  // Already checked this one
            }
            
            auto [search_seq_id, search_positions] = read_buffer_entry(buffer_index);
            
            if (search_seq_id == next_seq_id) {
                std::cout << currentTimeFormatted() 
                          << " [C++] Found policy action for seq_id " << search_seq_id 
                          << " at buffer index " << buffer_index << std::endl;
                return search_positions;
            }
        }
        
        // If we get here, the seq_id we need wasn't found in the buffer
        std::cout << currentTimeFormatted() 
                  << " [C++] Policy action not found for seq_id " << next_seq_id 
                  << " in buffer" << std::endl;
        return {};
        
    } catch (const std::exception& e) {
        std::cerr << currentTimeFormatted() 
                  << " [C++] Failed to read from policy target shared memory: " 
                  << e.what() << std::endl;
        return {};
    }
}
