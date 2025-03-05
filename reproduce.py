import os
import time
import numpy as np
import threading

# According to your project structure, import the relevant control classes:
# - RealTimeUR5Controller: Use servoL() to update the pose in real time
# - AsyncGripperController: Connect once, open/close the gripper multiple times
# - RealTimeUR5ControllerUsingMoveL: Use moveL() to update the pose in real time
# If they are in the same folder, you can use:
# from realtime_ur5_controller import RealTimeUR5Controller
# from async_gripper_controller import AsyncGripperController
# Otherwise, modify the actual path:
from ur5_controller import RealTimeUR5ControllerUsingMoveL
from ur5_controller import AsyncGripperController


def replay_dataset(
    data_folder="./dataset/data/task2",
    ur5_ip="192.168.1.60",
    replay_hz=5.0
):
    """
    This function reads the trajectory data (.npy) saved from previous collection from the specified directory (data_folder),
    and replays it on the UR5 robot in sequence.

    :param data_folder: The directory containing the collected data (e.g., targ0.npy, targ1.npy, ...)
    :param ur5_ip: The IP address of the UR5 control cabinet
    :param replay_hz: The replay frequency (how many steps per second), e.g., 5.0 means sending a new command every 0.2s
    """
    # 1. Initialize the robot and gripper controller
    ur5_controller = RealTimeUR5ControllerUsingMoveL(ur5_ip)
    gripper_controller = AsyncGripperController(ur5_ip)

    # If you want to open the gripper first, you can call:
    gripper_controller.control_gripper(close=False, force=100, speed=30)
    time.sleep(1.0)  # Wait for the gripper to open

    current_pose = [-0.3, -0.030, 0.320, np.pi, 0, 0]
    ur5_controller.send_pose_to_robot(current_pose)
    ur5_controller.start_realtime_control(frequency=5)


    # 2. List all 'targ*.npy' files in the directory and sort them by number
    npy_files = []
    for fname in os.listdir(data_folder):
        if fname.startswith("targ") and fname.endswith(".npy"):
            npy_files.append(fname)
    npy_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

    if not npy_files:
        print(f"[WARN] No 'targ*.npy' files found in {data_folder}.")
        return

    print(f"[INFO] Found {len(npy_files)} files to replay...")

    # 3. Start reading data at a fixed frequency (replay_hz) and sending it to the robot
    period = 1.0 / replay_hz

    try:
        step_count = 0
        for f in npy_files:
            file_path = os.path.join(data_folder, f)
            data = np.load(file_path, allow_pickle=True).item()

            # Common fields in the dataset: 'joint', 'pose', 'image', 'depth_image', 'gripper'
            pose = data.get('pose', None)
            gripper_val = data.get('gripper', 0)

            if pose is None:
                print(f"[ERROR] No 'pose' in file: {f}, skip.")
                continue

            # 4. Send the pose to the robot
            ur5_controller.send_pose_to_robot(pose)

            # 5. Send the gripper open/close command
            # If gripper_val == 1, it means closed; == 0, it means open (according to your data definition at the time)
            close_gripper = bool(gripper_val)
            gripper_controller.control_gripper(close=close_gripper, force=100, speed=80)

            print(f"[REPLAY] Step {step_count}, File={f}, Pose={pose}, Gripper={gripper_val}")
            step_count += 1

            # Wait for a control cycle
            time.sleep(period)

        print("[INFO] Finished replaying all data.")

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        # 6. Before ending, disconnect and release resources
        gripper_controller.disconnect()
        ur5_controller.close()
        pass


def main():
    # Configure parameters
    DATA_FOLDER = "./dataset/data/task2"        # The dataset directory, same as when collecting
    UR5_IP = "192.168.1.60"     # Your UR5 IP
    REPLAY_HZ = 5.0             # 5 steps per second, corresponding to 0.2s interval

    replay_dataset(
        data_folder=DATA_FOLDER,
        ur5_ip=UR5_IP,
        replay_hz=REPLAY_HZ
    )


if __name__ == "__main__":
    main()
