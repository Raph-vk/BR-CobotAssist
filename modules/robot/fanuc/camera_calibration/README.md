# Fanuc Robot Camera Calibration Documentation

This document explains the structure and contents of the calibration data collected using the Fanuc robot camera calibration scripts.

## Directory Structure

```
camera_calibration/
├── README.md                           # This documentation file
├── calibration/                       # Multi-position calibration data
│   └── YYYYMMDD_HHMMSS/               # Timestamped session directories
│       ├── calibration_pos_01.h5      # Position 1 data
│       ├── calibration_pos_02.h5      # Position 2 data
│       ├── ...                        # Additional positions
│       ├── calibration_pos_50.h5      # Position 50 data
│       └── calibration_data.json      # Session metadata
└── single_pos_calibration/            # Single-position calibration data
    └── YYYYMMDD_HHMMSS/               # Timestamped session directories
        ├── photo_01.h5                # Photo 1 data
        ├── photo_02.h5                # Photo 2 data
        ├── ...                        # Additional photos
        ├── photo_50.h5                # Photo 50 data
        └── calibration_data.json      # Session metadata
```

### Target Cameras
The system is configured to capture data from four Intel RealSense D405 cameras:
- **Camera 1 (cam_1)**: Serial `218622271425` *(EE Camera)*
- **Camera 2 (cam_2)**: Serial `218622271391` *(EE Camera)*
- **Camera 3 (cam_3)**: Serial `130322272626`
- **Camera 4 (cam_4)**: Serial `218622271529`

#### Camera Usage in Calibration:
- **EE Calibration** (`robot_fanuc_EEcam_calibration.py`): Uses only the 2 EE cameras (`218622271391`, `218622271425`)
- **Main Calibration** (`robot_fanuc_cam_calibration.py`): Uses 3 cameras at a time, executed twice:
  - Run 1: 2 EE cameras + cam_3 (`218622271391`, `218622271425`, `130322272626`)
  - Run 2: 2 EE cameras + cam_4 (`218622271391`, `218622271425`, `218622271529`)

### Image Specifications
- **Resolution**: 1280 × 720 pixels
- **Format**: BGR8 (3-channel color)
- **Compression**: GZIP compression (level 6) in HDF5 files
- **Frame Rate**: 30 FPS during capture

## HDF5 File Structure

Each HDF5 file contains comprehensive robot and camera data for a single capture instance.

### File Naming Convention
- **Multi-position**: `calibration_pos_XX.h5` (where XX = 01-50)
- **Single-position**: `photo_XX.h5` (where XX = 01-50)

### Internal HDF5 Structure

```
filename.h5
├── robot_state/                        # Robot state information
│   ├── joint_positions                 # Dataset: [J1, J2, J3, J4, J5, J6] in degrees
│   ├── cartesian_position/             # Group: End-effector pose
│   │   ├── position                    # Dataset: [x, y, z] in mm
│   │   ├── orientation                 # Dataset: [w, p, r] in degrees
│   │   ├── @x, @y, @z                  # Attributes: Individual position values
│   │   └── @w, @p, @r                  # Attributes: Individual orientation values
│   ├── @position_index                 # Attribute: Position/photo number
│   ├── @timestamp                      # Attribute: ISO timestamp
│   └── @gripper_state                  # Attribute: Gripper state (0/1)
├── cameras/                            # Camera data group
│   ├── camera_218622271391/            # Camera 1 data
│   │   ├── color_image                 # Dataset: (720, 1280, 3) uint8 array
│   │   ├── @serial_number              # Attribute: Camera serial number
│   │   ├── @color_timestamp            # Attribute: Capture timestamp
│   │   └── @color_resolution           # Attribute: [width, height]
│   ├── camera_218622271425/            # Camera 2 data (same structure)
│   └── camera_218622271529/            # Camera 3 data (same structure)
├── @created_timestamp                  # Attribute: File creation time
├── @position_index                     # Attribute: Position/photo index
├── @target_joint_position              # Attribute: Target joint angles
├── @num_cameras                        # Attribute: Number of cameras used
└── @camera_serial_numbers             # Attribute: List of camera serials
```

### Data Types and Sizes

#### Robot State Data
- **Joint Positions**: 6 × float64 (degrees)
- **Cartesian Position**: 3 × float64 (mm)
- **Cartesian Orientation**: 3 × float64 (degrees, WPR format)
- **Gripper State**: int (0 = open, 1 = closed)

#### Camera Data
- **Color Images**: 720 × 1280 × 3 × uint8 (≈2.76 MB per image, compressed)
- **Timestamps**: float64 (Unix timestamp)
- **Resolution**: 2 × int (width, height)

#### File Sizes
- **Per HDF5 file**: ~8-12 MB (3 compressed images + metadata)
- **Complete session**: ~400-600 MB (50 files)

## Coordinate Systems

### Joint Angles
- **Range**: Each joint ±5° from base position
- **Units**: Degrees
- **Convention**: Standard Fanuc joint convention

### Cartesian Coordinates
- **Position**: [x, y, z] in millimeters
- **Orientation**: [w, p, r] in degrees (Wrist-Pitch-Roll)
- **Reference Frame**: Robot base coordinate system

### Forward Kinematics
The system uses Denavit-Hartenberg parameters specific to the **LR Mate 200iD 7L** robot:
- Link parameters: a, d, α values for each joint
- Transformation matrices: 4×4 homogeneous transformations
- End-effector pose calculation from joint angles

## Recorded Episode Data Structure (HDF5 Files)

The TOS system saves robot demonstrations and data in HDF5 files located in `data/YYYYMMDD_HHMMSS/episode_X.hdf5`. Each file contains the following structure:

```
episode_X.hdf5
├── images/                              # Camera data
│   ├── cam_1/
│   │   ├── color                        # Shape: (N, 240, 424, 3), dtype: uint8
│   │   ├── color_timestamps             # Shape: (N,), dtype: float64
│   │   ├── depth                        # Shape: (N, 240, 424), dtype: uint16
│   │   └── depth_timestamps             # Shape: (N,), dtype: float64
│   ├── cam_2/                           # Same structure as cam_1
│   └── cam_3/                           # Same structure as cam_1
├── master_positions                     # Shape: (N, 7), dtype: float32
├── robot_positions                      # Shape: (N, 7), dtype: float32  
├── send_position_robots                 # Shape: (N, 7), dtype: float32
├── robot_position_timestamps            # Shape: (N,), dtype: float64
└── metadata/
    ├── cameras/
    │   ├── cam_1/                       # Camera calibration data
    │   ├── cam_2/
    │   └── cam_3/
    └── joint_states/                    # Robot configuration metadata
```

### Position Data Format
All position arrays contain **7 elements**: `[joint1, joint2, joint3, joint4, joint5, joint6, gripper_state]`

- **master_positions**: Commands from the operator (teachbot/master device)
  - Joints 1-6: Degrees (robot joint angles)
  - Element 7: Gripper state (0.0 = OFF, 1.0 = ON)

- **robot_positions**: Actual robot state feedback from Fanuc robot
  - Joints 1-6: Degrees (actual robot joint positions)  
  - Element 7: IO value (0.0 = gripper OFF, 1.0 = gripper ON)

- **send_position_robots**: Commands sent to robot after processing
  - Joints 1-6: Degrees (processed joint commands)
  - Element 7: Processed gripper state (0.0 = OFF, 1.0 = ON)

- **robot_position_timestamps**: Unix timestamps for each robot state sample

### Data Flow Summary
1. **Input**: 7-element arrays from master device (6 joints + gripper)
2. **Processing**: Ruckig trajectory planning on 6 robot joints only
3. **Robot Communication**: 6 joint positions + separate gripper ON/OFF flags  
4. **Feedback**: 7-element arrays from robot (6 joints + IO state)
5. **Storage**: All arrays saved as 7-element format for consistency

Where `N` = number of recorded timesteps (typically 313 for a ~12.5 second recording at 25 Hz)
