// File: cpp_test_pybind.hpp
#pragma once

#include "cpp_test.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(cpp_test_module, m) {
    py::class_<CppTest>(m, "CppTest")
        .def(py::init<
             bool, double, int,
             std::vector<double>,
             std::vector<std::array<double,3>>,
             bool, int,
             std::vector<std::string>,  // (8) server_address
             std::vector<std::string>,  // (9) robot_address
             int,                       // (10) check_queue_period_divisor
             bool,                      // (11) play_recording_active
             std::deque<std::vector<double>>, // (12) master_positions
             double, double, double,    // (13)(14)(15) gripper_treshold, gripper_delay, robot_speed
             std::vector<double>,       // (16) target_pos_received
             bool, bool,                // (17)(18) robot_running, recording
             std::vector<double>,       // (19) upper_limits
             std::vector<double>,       // (20) lower_limits
             bool, bool, bool,           // (21)(22)(23) gripper_on, gripper_off, gripper_state
             py::object,                // (24) python callback
             int,                       // (25) shm_capacity
             std::string,               // (26) shm_name
             int,                       // (27) shm_policy_capacity
             std::string,               // (28) shm_policy_name
             int,                       // (29) shm_policy_target_capacity
             std::string,               // (30) shm_policy_target_name
             std::string,               // (31) shm_policy_target_format
             int                        // (32) shm_policy_target_entry_size
        >())

        .def("start_control_loop", &CppTest::startControlLoop)
        .def("stop_control_loop", &CppTest::stopControlLoop,
             py::call_guard<py::gil_scoped_release>())

        // Python -> C++ bindings
        .def("update_target_position", &CppTest::updateTargetPosition)
        .def("set_recording", &CppTest::setRecording)
        .def("set_policy_active", &CppTest::setPolicyActive);
}
