import numpy as np
import cv2
import os
import time

def load_and_show_data(data_folder="data"):
    """
    依次读取 data_folder 下的 targ{i}.npy 文件，
    打印其中的 pose 并显示 primary_image。
    按键 'q' 退出循环。
    """
    i = 0  # 从第0个文件开始
    while True:
        # 构造文件名，如 data/targ0.npy, data/targ1.npy 等
        file_path = os.path.join(data_folder, f"targ{i}.npy")
        # 如果文件不存在，就停止
        if not os.path.exists(file_path):
            time.sleep(5)
            print(f"[INFO] 文件 {file_path} 不存在，读取结束。")
            break

        # 读取 .npy 文件
        data = np.load(file_path, allow_pickle=True).item()  # 这里要用 .item() 取出字典

        # 获取 primary_image 和 pose
        primary_image = data.get('primary_image', None)
        primary_image = primary_image[..., ::-1]  # BGR->RGB
        #  primary_image = primary_image[..., ::-1]  # BGR->RGB
        primary_image = cv2.resize(primary_image, (336, 336), interpolation=cv2.INTER_LINEAR)
        pose = data.get('pose', None)
        gripper_val = data.get('gripper', None)
        secondary_image = data.get('secondary_image', None)
        wrist_image = data.get('wrist_image', None)
        dep_primary_image = data.get('primary_depth_image', None)

        # 打印关键信息
        print(f"当前索引: {i}")
        if pose is not None:
            print("Pose 数据:", pose)  # 形如 [x, y, z, Rx, Ry, Rz]
        else:
            print("[WARNING] 当前文件中不存在 'pose' 数据。")
        if gripper_val is not None:
            print("Gripper 数据:", gripper_val)
        else:
            print("[WARNING] 当前文件中不存在 'gripper' 数据。")
        if gripper_val > 0.5:
            time.sleep(0.2)
        # 显示 primary_image
        if wrist_image is not None:
            cv2.imshow("Primary Image", primary_image)
        else:

            print("[WARNING] 当前文件中不存在 'primary_image' 数据。")

        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting camera loop.")
            break
        time.sleep(0.1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 示例：指定保存数据的文件夹
    load_and_show_data(data_folder="data_clean/task33")
