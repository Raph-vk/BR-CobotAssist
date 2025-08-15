# TOS Network Architecture & IP Distribution

## Overview

TOS uses a structured network architecture with dedicated subnets for different types of communication. This ensures reliable separation between internet access, inter-PC communication, robot control, and camera networks.

## Network Topology

```
                                Internet Access (Customer-provided)
                                                |
                                             [IP ???]
                                           [UI Host PC]
                                                |
                                192.168.100.x (General Communication Subnet)
                                                |
        ┌───────────────────┬───────────────────┬───────────────────┬───────────────────┐
        │      Setup 1      │      Setup 2      │      Setup N      │     Accessories   │
        │  192.168.100.101  │  192.168.100.102  │  192.168.100.103+ │  192.168.100.151+ │
        │    (UI Host)      │   (Robot Ctrl)    │   (Robot Ctrl)    │   (Teachbots,etc) │
        └───────────────────┴───────────────────┴───────────────────┴───────────────────┘
                  |                   |                   |
           192.168.150.101     192.168.150.101     192.168.150.101
           (Robot Control)     (Robot Control)     (Robot Control)
                  |                   |                   |
        ┌─────────────────┐ ┌─────────────────┐  ┌─────────────────┐
        │    Robot 1      │ │    Robot 2      │  │    Robot N      │
        │ 192.168.150.151 │ │ 192.168.150.152 │  │ 192.168.150.153+│
        │ (FANUC LRMate)  │ │ (Techman TM12)  │  │ (Model Based)   │
        └─────────────────┘ └─────────────────┘  └─────────────────┘

        Camera Networks (Per-Camera Subnets):

        PC: 192.168.200.101         PC: 192.168.201.101      PC: 192.168.202+.101
                 |                          |                         |
        ┌─────────────────┐        ┌─────────────────┐       ┌─────────────────┐
        │    Camera 1     │        │    Camera 2     │       │    Camera N     │
        │ 192.168.200.151 │        │ 192.168.201.151 │       │ 192.168.202.151 │
        └─────────────────┘        └─────────────────┘       └─────────────────┘
                 |                          |                         |
          192.168.200.101            192.168.201.101           192.168.202+.101
         (Camera Subnet 1)          (Camera Subnet 2)         (Camera Subnet 3)
```

## Subnet Allocation

### 192.168.100.x - General Communication Subnet
**Purpose**: Inter-PC communication, RabbitMQ messaging, TOS UI access

| IP Range | Purpose | Description |
|----------|---------|-------------|
| `192.168.100.101` | Setup 1 PC | First PC |
| `192.168.100.102` | Setup 2 PC | Second PC |
| `192.168.100.103+` | Setup N PC | Additional PCs |
| `192.168.100.151` | Teachbot | TOS Teachbot Cobotlayout 200-200 mm |
| `192.168.100.152+` | Accessories | Network accessories, teachbots, etc. |

### 192.168.150.x - Robot Control Subnet
**Purpose**: Direct robot communication and control. The Setup PC IP adress will always be 192.168.150.101

**IP Assignment**: Based on robot brand and model (not setup number)

| IP Address | Robot Brand | Robot Model | Assignment |
|------------|-------------|-------------|------------|
| `192.168.150.101` | N/A | PC Controller | Always the controlling PC |
| `192.168.150.151` | FANUC | LR Mate 200iD 7L | Primary FANUC robot |
| `192.168.150.152` | Techman | TM12 | Techman collaborative robot |
| `192.168.150.153+` | [Future] | [Future Models] | Reserved for expansion |

**Note**: Multiple robots of the same model receive sequential IP assignments if two are ever to be ocnnected to one robot controller (e.g., Primary at .151, Secondary at .155)

### 192.168.200+.101 - Camera Subnets
**Purpose**: Ethernet camera communication

**Camera Subnet Assignment**: Each camera gets its own dedicated subnet on each pc

| Subnet | PC IP | Camera IP | Assignment |
|--------|-------|-----------|------------|
| `192.168.200.x` | `192.168.200.101` | `192.168.200.151` | Camera 1 network |
| `192.168.201.x` | `192.168.201.101` | `192.168.201.151` | Camera 2 network |
| `192.168.202.x` | `192.168.202.101` | `192.168.202.151` | Camera 3 network |
| `192.168.20N.x` | `192.168.20N.101` | `192.168.20N.151` | Camera N network |

**Note**: Each camera requires a dedicated subnet with individual PC-to-camera connection

## Configuration Rules

### Setup Numbering
- **Setup ID**: Sequential numbering starting from 1
- **General Comm IP**: `192.168.100.10{setup_id}`
- **Robot Control IP**: Always `192.168.150.101` (PC controller)
- **Robot IP**: `192.168.150.151+` (assigned by robot model/brand, not setup)

### Network Isolation
- **Robot Control**: PCs connect to robots on the 192.168.150.x subnet (robot IPs assigned by model/brand)
- **General Communication**: All PCs communicate with the UI host via the 192.168.100.x subnet
- **Camera Networks**: Each camera has its own isolated subnet (192.168.20N.x) with dedicated PC connection

### Network Adapter Configuration
- **UI Host PC**: Requires multiple network adapters:
  - Customer internet connection (dynamic IP)
  - General communication subnet (192.168.100.101)
  - Robot control subnet (192.168.150.101) if robots connected
  - Camera subnets (192.168.20N.101) if cameras connected
- **Robot Controller PCs**: Typically dual adapters:
  - General communication subnet (192.168.100.10X)
  - Robot control subnet (192.168.150.101)
  - Camera subnets (192.168.20N.101) if cameras connected

### RabbitMQ Communication
- **Message Broker**: Runs on UI host PC (192.168.100.101)
- **Connection**: All robot controller PCs connect to UI host for messaging
- **Dynamic Resolution**: Uses `"auto"` host setting to automatically resolve UI host IP

## Current Implementation Example

| Component | IP Address | Subnet | Role | Connections |
|-----------|------------|---------|------|-------------|
| UI Host PC (Setup 1) | `192.168.100.101` | General Comm | TOS UI + RabbitMQ + Robot Controller | 4 connections: Internet (customer), Setup 2 PC, FANUC robot, TOS Teachbot |
| Robot Controller PC (Setup 2) | `192.168.100.102` | General Comm | Robot Controller Only | 2 connections: UI Host PC (RabbitMQ), FANUC robot |
| FANUC LR Mate 200iD 7L | `192.168.150.151` | Robot Control | Connected to Setup 2 | 1 connection: Setup 2 PC |
| TOS Teachbot | `192.168.100.151` | General Comm | Cobotlayout 200-200 mm | 1 connection: UI Host PC |

---

## Network Implementation Notes

- **Robot IP Assignment**: IPs are assigned based on robot brand and model, allowing for consistent addressing across different setups
- **Multi-Robot Support**: A single PC can control multiple robots, each with its own model-based IP assignment  
- **Camera Isolation**: Each camera requires its own subnet for optimal performance and isolation
- **Scalable Architecture**: The subnet structure supports expansion without IP conflicts
- **Flexible Connectivity**: UI host can support multiple network connections as needed for robots and cameras

