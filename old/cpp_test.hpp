#pragma once
// 1) pybind before use of pybind11::object
#include <pybind11/pybind11.h>
namespace py = pybind11;

// STL / POSIX
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// Ruckig
#include <ruckig/ruckig.hpp>

/*────────────────────────────  Shared capacity calculation  ───────────────────────────*/
// Calculate shared memory buffer capacity using deterministic formula.
// This ensures Python and C++ use the same capacity regardless of config.
inline int calculate_shm_capacity(double record_duration, double control_dt) {
    int size = static_cast<int>(record_duration / control_dt);
    // Round size up to nearest power of 2 for efficient wraparound
    int capacity = 1;
    while (capacity < size) {
        capacity <<= 1;
    }
    return capacity;
}

/*────────────────────────────  Ring-buffer definitions  ───────────────────────────*/
// Maximum DOF supported (can be configured down from Python)
constexpr std::size_t kMaxDof   = 8;                    // Maximum robot DOF supported
// These will be configured from Python at runtime
// constexpr std::size_t kCapacity = 2048;               // will be runtime parameter  
// constexpr const char* kShmName  = "/fanuc_joint_data"; // will be runtime parameter

#pragma pack(push, 1)
struct JointDataSlot {
    double  master_pos[kMaxDof];       // Target position (inp.target_position)
    double  send_position_robot[kMaxDof];  // Position sent to robot (out.new_position)
    double  robot_pos[kMaxDof];        // Actual robot position received
    uint8_t gripper_on;
    uint8_t gripper_off;
    uint32_t seq_id;
    double  timestamp;
    double  robot_position_timestamp;  // Timestamp when robot position was received
};
#pragma pack(pop)

struct RingBufferHeader {
    std::atomic<uint32_t> write_idx{0};
    std::atomic<uint32_t> read_idx {0};
    uint32_t capacity  {0};        // Will be set at runtime
    uint32_t slot_size {sizeof(JointDataSlot)};
};

/*────────────────────────────  Minimal SPSC writer  ───────────────────────────────*/
class ShmRingBufferWriter {
public:
    ShmRingBufferWriter(int capacity, const std::string& shm_name, int dof);  // configurable parameters
    ~ShmRingBufferWriter();                // munmap + close (no unlink)
    void push(const JointDataSlot& s, int dof);     // lock-free single-producer push
private:
    int                fd_   {-1};
    std::size_t        bytes_{0};
    RingBufferHeader*  hdr_  {nullptr};
    JointDataSlot*     slots_{nullptr};
    int                capacity_;    // Runtime capacity
    std::string        shm_name_;    // Runtime shared memory name
};

/*─────────────────────────────────  Main class  ───────────────────────────────────*/
class CppTest {
public:
    CppTest(bool started_streaming, double control_dt, int dof,
            const std::vector<double>& start_position,
            const std::vector<std::array<double,3>>& joint_limits,
            bool joint_synchronization, int action_buffer_length,
            const std::vector<std::string>& server_address,
            const std::vector<std::string>& robot_address,
            int check_queue_period_divisor, bool play_recording_active,
            const std::deque<std::vector<double>>& master_positions,
            double gripper_treshold, double gripper_delay, double robot_speed,
            const std::vector<double>& target_pos_received, bool robot_running,
            bool recording, const std::vector<double>& upper_limits,
            const std::vector<double>& lower_limits,
            bool gripper_on, bool gripper_off, bool gripper_state,
            py::object py_callback,
            // Shared memory configuration parameters
            int shm_capacity, const std::string& shm_name,
            int shm_policy_capacity, const std::string& shm_policy_name,
            // Policy target shared memory parameters
            int shm_policy_target_capacity, const std::string& shm_policy_target_name,
            const std::string& shm_policy_target_format, int shm_policy_target_entry_size);
    ~CppTest();

    void startControlLoop(bool started_streaming);
    void stopControlLoop();
    void updateTargetPosition(const std::vector<double>& t) { target_pos_received_ = t; }
    void setRecording(bool rec);
    void setPolicyActive(bool active) { run_policy_active_ = active; }

private:
    /* threads */
    void controlLoop(bool started_streaming);
    void receiveLoop();
    void startReceiving();
    void stopReceiving();

    /* motion helpers */
    void setupRuckig();
    std::vector<double> updateRuckigInput(const std::vector<double>& action_master,
                                          ruckig::InputParameter<0>& inp,
                                          const std::vector<double>& previous);
    bool trajectoryCalculation(ruckig::Ruckig<0>& otg,
                               ruckig::InputParameter<0>& inp,
                               ruckig::OutputParameter<0>& out,
                               const std::vector<double>& current_position);
    bool handleTrajectoryError(const std::exception& e,
                               ruckig::InputParameter<0>& inp,
                               ruckig::OutputParameter<0>& out,
                               ruckig::Ruckig<0>& otg, int dof);
    bool stopRobot(ruckig::Ruckig<0>& otg,
                   ruckig::InputParameter<0>& inp,
                   ruckig::OutputParameter<0>& out);

    /* I/O */
    bool sendJointPos(const std::vector<double>& position,
                      bool gripper_on, bool gripper_off);
    void decodeStatusPacket(const uint8_t* data, size_t size);
    void determineGripperState(double gripper_state_val);

    /* misc helpers */
    std::vector<double> roundPosition(const std::vector<double>& position);
    std::vector<double> J3Interaction(const std::vector<double>& a);
    std::vector<double> J3InteractionRev(const std::vector<double>& a);
    double nowSec() const;  // helper to get current time in seconds

    /* new: write current sample to shared memory */
    void publishSampleToShm(const std::vector<double>& master_position,
                            const std::vector<double>& send_position_robot,
                            const std::vector<double>& robot_position,
                            double robot_position_timestamp,
                            bool g_on, bool g_off);
    
    /* write current sample to policy shared memory */
    void publishSampleToPolicyShm(const std::vector<double>& master_position,
                                  const std::vector<double>& send_position_robot,
                                  const std::vector<double>& robot_position,
                                  double robot_position_timestamp,
                                  bool g_on, bool g_off);
    
    /* get next policy action from shared memory */
    std::vector<double> getNextPolicyAction();

    /* networking */
    std::string server_ip_; uint16_t server_port_{0};
    std::string robot_ip_;  uint16_t robot_port_{0};
    int udp_socket_fd_{-1};
    std::atomic<bool> udp_running_{false};
    std::thread udp_thread_;

    /* control-thread */
    std::thread control_thread_;
    std::atomic<bool> robot_running_{false};

    /* state from robot */
    std::atomic<uint32_t> seq_id_received_{0};
    uint32_t seq_id_sent_{0};
    std::vector<double> joint_state_received_;
    double joint_state_received_time_{0.0};
    bool started_receiving_motion_stream_{false};

    /* config / runtime parameters */
    bool   started_streaming_;
    double control_dt_;
    int    dof_;
    std::vector<double> start_position_;
    std::vector<std::array<double,3>> joint_limits_;
    bool   joint_synchronization_;
    int    action_buffer_length_;
    int    check_queue_period_divisor_;
    bool   play_recording_active_;
    std::deque<std::vector<double>> master_positions_;
    double gripper_treshold_;
    double gripper_delay_;
    double robot_speed_;
    std::vector<double> target_pos_received_;
    bool   recording_;
    bool   run_policy_active_{false};  // Flag to indicate if policy is providing targets
    std::vector<double> upper_limits_, lower_limits_;
    bool   gripper_on_, gripper_off_, gripper_state_;
    double time_last_sent_{0.0};
    double gripper_state_change_time_{0.0};
    double gripper_state_change_time_threshold_{0.0};

    /* ruckig */
    std::unique_ptr<ruckig::Ruckig<0>>          otg_;
    std::unique_ptr<ruckig::InputParameter<0>>  inp_;
    std::unique_ptr<ruckig::OutputParameter<0>> out_;
    int error_3_occur_{0};

    /* python callback */
    py::object py_callback_;

    /* shared memory configuration (from Python) */
    int    shm_capacity_;
    std::string shm_name_;
    int    actual_dof_;  // Actual DOF being used (≤ kMaxDof)
    // Policy interface shared memory configuration
    int    shm_policy_capacity_;
    std::string shm_policy_name_;
    // Policy target shared memory configuration  
    int    shm_policy_target_capacity_;
    std::string shm_policy_target_name_;
    std::string shm_policy_target_format_;
    int    shm_policy_target_entry_size_;

    /* shared-memory writers */
    std::unique_ptr<ShmRingBufferWriter> shm_writer_;       // For save interface (recording)
    std::unique_ptr<ShmRingBufferWriter> shm_policy_writer_; // For policy interface (real-time)
    
    /* policy target shared memory reader */
    int shm_policy_target_fd_{-1};
    void* shm_policy_target_ptr_{nullptr};
    size_t shm_policy_target_size_{0};
};
