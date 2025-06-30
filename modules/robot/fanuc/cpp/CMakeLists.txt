cmake_minimum_required(VERSION 3.14)
project(fanuc_control_loop_module)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include_directories(${Python3_INCLUDE_DIRS})

find_package(pybind11 REQUIRED)

# Manual find library or if you have a system-wide lib (as you do):
find_library(RUCKIG_LIBRARY NAMES ruckig HINTS /usr/local/lib /usr/lib /opt/ros/noetic/lib)
if(NOT RUCKIG_LIBRARY)
    message(FATAL_ERROR "Could not find libruckig.so!")
endif()
message(STATUS "Found ruckig library at ${RUCKIG_LIBRARY}")

include_directories("/usr/local/include")

set(SOURCES
    control_loop.cpp
    control_loop_pybind.cpp
)

pybind11_add_module(control_loop_module
    ${SOURCES}
)

target_link_libraries(control_loop_module
    PRIVATE
        ${RUCKIG_LIBRARY}
        pthread
        rt
        ${Python3_LIBRARIES}
)

set_target_properties(control_loop_module PROPERTIES
    OUTPUT_NAME "control_loop_module"
    PREFIX ""
)

message(STATUS "CMake script configured. Ready to build the Pybind module (C++17).")

