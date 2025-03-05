
# UR5 Robotic Control and Dataset Generation System

This project is used to control the UR5 robot and generate the dataset for the robot operation. Special for the UR5 robot, used for OpenVLA fine-tuning.

## Environment Setup

### Prerequisites
- Python 3.10
- pyrealsense2>=2.55.1.6486
- opencv-python
- ur_rtde>=1.5.9
- numpy==1.24.3
- pyspacemouse


### Installation
```bash
# suggest using conda to install the packages
conda create -n ur5_controller python=3.10
conda activate ur5_controller

# Install pyspacemouse
According to the official website: https://github.com/JakubAndrysek/PySpaceMouse   
Possible instuctions:
https://blog.csdn.net/qq_40081208/article/details/137675822
https://bbs.archlinux.org/viewtopic.php?id=278341


# Install required Python packages
pip install -r requirements.txt

```

## Project Structure
```
├── ur5_controller/
│   ├── get_pos_revised_class.py       # Main control logic
│   ├── vacuum_gripper.py              # Gripper control interface
│   ├── get_pos_revised_class_moveL.py  # Main control logic for moveL
│   └── __init__.py                    # Package exports
├── ur5_robo_dataset_dataset_builder.py # Dataset generation pipeline
└── readme.md                          # This documentation
```


## Basic Usage
Start the robot control script and save the data to the dataset folder.
```bash
./ur5_controller/get_pos_revised_class_moveL.py
```
clean the dataset folder.
```bash
./filter_dataset.py
```
build the dataset.
```bash
tfds build
```
## Data Collection Protocol

### Raw Dataset Structure
```
data/
├── task0/
│   ├── target0.npy
│   ├── target1.npy
│   ├── target2.npy
│   └── ...
├── task1/
│   ├── target0.npy
│   ├── target1.npy
│   ├── target2.npy
│   └── ...
├── task2/
└── ...
```

### Raw Data Fields (per step)
- RGB images (3x640x480)
- Depth maps (1x640x480)
- End-effector pose (6D vector)
- Joint positions (6D vector)
- Gripper state (0 or 1)

### Player:
uesd to play the raw data

### vidio_gen:
used to generate the video from the raw data
