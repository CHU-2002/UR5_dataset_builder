import numpy as np
import cv2
import os
import time

def load_and_save_video(data_folder="data", output_video_path="output.mp4"):
    """
    依次读取 data_folder 下的 targ{i}.npy 文件，
    打印其中的 pose / gripper 等信息，并将 primary_image 写入视频文件。
    """
    # 定义视频编码和输出文件
    fps = 20  # 可以根据需要自行调整帧率
    frame_size = (640, 480)  # 统一视频帧大小
    
    # 使用 mp4v 编码器，保证兼容性
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    i = 0  # 从第0个文件开始
    while True:
        # 构造文件名
        file_path = os.path.join(data_folder, f"targ{i}.npy")

        # 如果文件不存在，就停止
        if not os.path.exists(file_path):
            time.sleep(5)
            print(f"[INFO] 文件 {file_path} 不存在，读取结束。")
            break

        # 读取 .npy 文件
        data = np.load(file_path, allow_pickle=True).item()

        # 获取 primary_image, pose, gripper 等信息
        primary_image = data.get('primary_image', None)
        pose = data.get('pose', None)
        gripper_val = data.get('gripper', None)

        # 打印关键信息
        print(f"\n当前索引: {i}")
        if pose is not None:
            print("Pose 数据:", pose)  # 形如 [x, y, z, Rx, Ry, Rz]
        else:
            print("[WARNING] 当前文件中不存在 'pose' 数据。")

        if gripper_val is not None:
            print("Gripper 数据:", gripper_val)
        else:
            print("[WARNING] 当前文件中不存在 'gripper' 数据。")

        # 处理 primary_image
        if primary_image is not None:
            print(f"[DEBUG] 原始 primary_image shape: {primary_image.shape}, dtype: {primary_image.dtype}")

            # 确保 primary_image 数据类型正确
            if primary_image.dtype != np.uint8:
                primary_image = (primary_image * 255).astype(np.uint8)

            # 确保 primary_image 是 3 通道
            if len(primary_image.shape) == 2:  # 灰度图
                primary_image = cv2.cvtColor(primary_image, cv2.COLOR_GRAY2BGR)

            elif primary_image.shape[2] == 4:  # 可能是 RGBA，需要转换为 BGR
                primary_image = cv2.cvtColor(primary_image, cv2.COLOR_RGBA2BGR)

            # 调整图像尺寸至 (336, 336)
            primary_image_resized = cv2.resize(primary_image, frame_size, interpolation=cv2.INTER_LINEAR)

            # OpenCV 需要 BGR 格式，所以不需要转换 RGB
            print(f"[DEBUG] 处理后 primary_image shape: {primary_image_resized.shape}, dtype: {primary_image_resized.dtype}")

            # 将图像写入视频
            out.write(primary_image_resized)

        else:
            print("[WARNING] 当前文件中不存在 'primary_image' 数据。")

        i += 1
        time.sleep(0.05)

    # 释放视频写入对象
    out.release()
    print(f"[INFO] 视频已保存至: {output_video_path}")

if __name__ == "__main__":
    # 运行代码
    load_and_save_video(data_folder="results/task14", output_video_path="output.mp4")
