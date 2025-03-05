import os
import numpy as np
import re
def _generate_examples(data_dir):
    """遍历 data_dir，生成训练数据"""
    print(f"Scanning dataset directory: {data_dir}")

    task_folders = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    
    if not task_folders:
        raise ValueError(f"No task folders found in {data_dir}")

    for episode_id, task_name in enumerate(task_folders):
        folder_path = os.path.join(data_dir, task_name)
        print(f"Processing task: {task_name}")
        npy_files = [
                f for f in os.listdir(folder_path)
                if f.endswith('.npy') and f.startswith('targ')
            ]
            # 按数字顺序排序文件名
        npy_files.sort(key=lambda fname: int(re.search(r'targ(\d+)\.npy', fname).group(1)))


        if not npy_files:
            print(f"No .npy files found in {folder_path}, skipping.")
            continue

        steps_data = []

        for npy_name in npy_files:
            npy_path = os.path.join(folder_path, npy_name)
            print(f"Loading file: {npy_path}")

            try:
                data_dict = np.load(npy_path, allow_pickle=True).item()
            except Exception as e:
                print(f"Error loading {npy_path}: {e}")
                continue

            step = {
                'observation': {
                    'primary_image': data_dict.get('primary_image', None),
                    'secondary_image': data_dict.get('secondary_image', None),
                    'wrist_image': data_dict.get('wrist_image', None),
                    'primary_depth': data_dict.get('primary_depth_image', None),
                    'secondary_depth': data_dict.get('secondary_depth_image', None),
                    'wrist_depth': data_dict.get('wrist_depth_image', None),
                    'pose': data_dict.get('pose', np.zeros(6, dtype=np.float32)).astype(np.float32),
                    'joint': data_dict.get('joint', np.zeros(6, dtype=np.float32)).astype(np.float32),
                    'gripper': float(data_dict.get('gripper', 0)),
                },
                'discount': 1.0,
                'reward': float(npy_name == npy_files[-1]),
                'is_first': float(npy_name == npy_files[0]),
                'is_last': float(npy_name == npy_files[-1]),
                'is_terminal': float(npy_name == npy_files[-1]),
                'language_instruction': "Put the white cube into the blue box",
            }
            steps_data.append(step)

        print(f"Generated episode {episode_id} with {len(steps_data)} steps.")
        yield episode_id, {
            'steps': steps_data,
            'episode_metadata': {
                'task_name': task_name,
            }
        }
if __name__ == '__main__':
    for idx, data in _generate_examples('data'):
        print(idx, data)
