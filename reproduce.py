import os
import time
import numpy as np
import threading

# 根据你的项目结构，导入相关控制类:
# - RealTimeUR5Controller: 用 servoL() 来实时更新姿态
# - AsyncGripperController: 一次连接，多次反复开/合爪子
# 如果你在同文件夹，则可以用:
# from realtime_ur5_controller import RealTimeUR5Controller
# from async_gripper_controller import AsyncGripperController
# 否则根据实际路径修改：
from serl_robot_infra.robot_controllers.get_pos_revised_class_movel import RealTimeUR5ControllerUsingMoveL
from serl_robot_infra.robot_controllers.get_pos_revised_class_movel import AsyncGripperController


def replay_dataset(
    data_folder="./dataset/data/task2",
    ur5_ip="192.168.1.60",
    replay_hz=5.0
):
    """
    该函数从指定目录 (data_folder) 中读取以往采集保存的轨迹数据 (.npy)，
    并按照顺序重放到 UR5 机器人上。

    :param data_folder: 存放采集数据 (如 targ0.npy, targ1.npy, ...) 的目录
    :param ur5_ip: UR5 控制柜的 IP
    :param replay_hz: 重放频率 (每秒多少步)，如 5.0 即每 0.2s 发送一次新指令
    """
    # 1. 初始化机器人与夹爪控制器
    ur5_controller = RealTimeUR5ControllerUsingMoveL(ur5_ip)
    gripper_controller = AsyncGripperController(ur5_ip)

    # 如果你想先让爪子张开，可以调用:
    gripper_controller.control_gripper(close=False, force=100, speed=30)
    time.sleep(1.0)  # 等待爪子张开完成

    current_pose = [-0.3, -0.030, 0.320, np.pi, 0, 0]
    ur5_controller.send_pose_to_robot(current_pose)
    ur5_controller.start_realtime_control(frequency=5)


    # 2. 列出该目录下所有 'targ*.npy' 文件，并按数字顺序排序
    npy_files = []
    for fname in os.listdir(data_folder):
        if fname.startswith("targ") and fname.endswith(".npy"):
            npy_files.append(fname)
    npy_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

    if not npy_files:
        print(f"[WARN] No 'targ*.npy' files found in {data_folder}.")
        return

    print(f"[INFO] Found {len(npy_files)} files to replay...")

    # 3. 开始按照固定频率（replay_hz）读数据并发送给机器人
    period = 1.0 / replay_hz

    try:
        step_count = 0
        for f in npy_files:
            file_path = os.path.join(data_folder, f)
            data = np.load(file_path, allow_pickle=True).item()

            # 数据集中常见字段: 'joint', 'pose', 'image', 'depth_image', 'gripper'
            pose = data.get('pose', None)
            gripper_val = data.get('gripper', 0)

            if pose is None:
                print(f"[ERROR] No 'pose' in file: {f}, skip.")
                continue

            # 4. 发送 pose 给机器人
            ur5_controller.send_pose_to_robot(pose)

            # 5. 发送爪子开合
            # 如果 gripper_val == 1, 表示闭合； == 0, 表示张开 (可根据你当时的数据定义)
            close_gripper = bool(gripper_val)
            gripper_controller.control_gripper(close=close_gripper, force=100, speed=80)

            print(f"[REPLAY] Step {step_count}, File={f}, Pose={pose}, Gripper={gripper_val}")
            step_count += 1

            # 等待一个控制周期
            time.sleep(period)

        print("[INFO] Finished replaying all data.")

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        # 6. 结束前，断开并释放资源
        gripper_controller.disconnect()
        ur5_controller.close()
        pass


def main():
    # 配置参数
    DATA_FOLDER = "./dataset/data/task2"        # 数据集目录，与之前采集时相同
    UR5_IP = "192.168.1.60"     # 你的 UR5 IP
    REPLAY_HZ = 5.0             # 每秒 5 步，对应 0.2s 间隔

    replay_dataset(
        data_folder=DATA_FOLDER,
        ur5_ip=UR5_IP,
        replay_hz=REPLAY_HZ
    )


if __name__ == "__main__":
    main()
