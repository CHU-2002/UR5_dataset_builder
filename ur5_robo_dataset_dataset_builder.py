


from typing import Iterator, Tuple, Any
import os
import re
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import cv2

class Ur5RoboDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for UR5 robot dataset with preprocessing."""

    VERSION = tfds.core.Version('1.1.3')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
        '1.0.1': 'Rename dataset & delete the empty tasks.',
        '1.0.2': 'Remove small/same steps',
        '1.1.0': '10 episodes demo, resized 224*224',
        '1.1.1': 'All episodes, resized 336*336',
        '1.1.2': 'Fix the bug of action calculation.',
        '1.1.3': 'Fix the BRG format bug.',
    }

    MAX_RES = 336  # 目标分辨率

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """返回数据集的元信息 (DatasetInfo)。"""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        # RGB 图像特征
                        **{
                            key: tfds.features.Image(
                                shape=(self.MAX_RES, self.MAX_RES, 3),
                                dtype=np.uint8,
                                encoding_format='jpeg',
                                doc=f'{key} camera RGB observation.'
                            ) for key in ["primary_image", "secondary_image", "wrist_image"]
                        },
                        # 深度图像特征
                        **{
                            key: tfds.features.Image(
                                shape=(self.MAX_RES, self.MAX_RES, 1),
                                dtype=np.uint16,
                                encoding_format='png',
                                doc=f'{key} depth observation.'
                            ) for key in ["primary_depth", "secondary_depth", "wrist_depth"]
                        },
                        # 机械臂相关特征
                        **{
                            'gripper': tfds.features.Scalar(
                                dtype=np.float32, 
                                doc='Gripper position.'
                            ),
                            'joint': tfds.features.Tensor(
                                shape=(6,), 
                                dtype=np.float32, 
                                doc='Joint angles (6D).'
                            ),
                            'pose': tfds.features.Tensor(
                                shape=(6,), 
                                dtype=np.float32, 
                                doc='End-effector pose (6D).'
                            ),
                        }
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,), 
                        dtype=np.float32, 
                        doc='7D action vector.'
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32, 
                        doc='Discount factor.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32, 
                        doc='Reward signal.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_, 
                        doc='Is first step of episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_, 
                        doc='Is last step of episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_, 
                        doc='Is terminal step.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Task instruction in text.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,), 
                        dtype=np.float32, 
                        doc='Task embedding.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Original data file path.'
                    )
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(data_dir='data_clean')
        }

    def _generate_examples(self, data_dir):
        """遍历 data_dir，生成训练数据"""
        #print(f"Scanning dataset directory: {data_dir}")

        task_folders = sorted(
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        )
        if not task_folders:
            raise ValueError(f"No task folders found in {data_dir}")

        for episode_id, task_name in enumerate(task_folders):
            folder_path = os.path.join(data_dir, task_name)
            #print(f"Processing task: {task_name}")
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
            last_state = None
            for npy_name in npy_files:
                npy_path = os.path.join(folder_path, npy_name)
                #print(f"Loading file: {npy_path}")

                try:
                    data_dict = np.load(npy_path, allow_pickle=True).item()
                except Exception as e:
                    print(f"Error loading {npy_path}: {e}")
                    continue
                

                pos_p = data_dict.get('pose', np.zeros(6, dtype=np.float32))
                gripper_val = data_dict.get('gripper', 0)
                state = np.concatenate([pos_p, [gripper_val]]).astype(np.float32)

            # 计算 action: 默认初始化为 7 维全 0
                action = np.zeros((7,), dtype=np.float32)

            # 这里可以根据 j==0 还是 j>0 来决定是否计算 action
                if npy_name == npy_files[0]:
                    # 第一个 step 通常没有上一个状态
                    pass
                else:
                    # action = current_state - last_state
                    action = (state - last_state).astype(np.float32)

                    # gripper 的绝对值大于 0.1 就表示开合
                    if gripper_val > 0.1:
                        action[6] = 1
                    else:
                        action[6] = 0

                last_state = state.copy()



                def process_img(img):
                        """对 RGB 图像进行 resize，并保证返回 (H, W, 3)。"""
                        if img is None:
                            return None
                        # 如果原图是 (H, W) 或 (H, W, 1) 需要先转成三通道
                        if img.ndim == 2:
                            # 灰度转伪 RGB
                            img = np.stack([img, img, img], axis=-1)
                        elif img.ndim == 3 and img.shape[-1] == 1:
                            # 同样视为灰度
                            img = np.concatenate([img, img, img], axis=-1)
                        # BGR -> RGB
                        img = img[..., ::-1]  # BGR->RGB
                        img = cv2.resize(img, (self.MAX_RES, self.MAX_RES))  # -> (H, W, 3)
                        # 如果你想 BGR -> RGB 可以用 img = img[..., ::-1]
                        return img.astype(np.uint8)

                def process_depth(depth_img):
                    """对深度图进行 resize，返回 (H, W, 1)"""
                    if depth_img is None:
                        return None
                    # 先 squeeze 到 (H, W)
                    if depth_img.ndim == 3 and depth_img.shape[-1] == 1:
                        depth_img = np.squeeze(depth_img, axis=-1)
                    # resize
                    depth_img = cv2.resize(depth_img, (self.MAX_RES, self.MAX_RES))
                    # 扩展成 (H, W, 1)
                    depth_img = depth_img[..., np.newaxis].astype(np.uint16)
                    return depth_img

                step = {
                    'observation': {
                        'primary_image': process_img(data_dict.get('primary_image', None)),
                        'secondary_image': process_img(data_dict.get('secondary_image', None)),
                        'wrist_image': process_img(data_dict.get('wrist_image', None)),
                        'primary_depth': process_depth(data_dict.get('primary_depth_image', None)[..., np.newaxis]),
                        'secondary_depth': process_depth(data_dict.get('secondary_depth_image', None)[..., np.newaxis]),
                        'wrist_depth': process_depth(data_dict.get('wrist_depth_image', None)[..., np.newaxis]),
                        'pose': data_dict.get('pose', np.zeros(6, dtype=np.float32)).astype(np.float32),
                        'joint': data_dict.get('joint', np.zeros(6, dtype=np.float32)).astype(np.float32),
                        'gripper': float(data_dict.get('gripper', 0)),
                    },
                    'action': action,
                    'discount': 1.0,
                    'reward': float(npy_name == npy_files[-1]),
                    'is_first': float(npy_name == npy_files[0]),
                    'is_last': float(npy_name == npy_files[-1]),
                    'is_terminal': float(npy_name == npy_files[-1]),
                    'language_instruction': "Put the white cube into the blue box",
                    'language_embedding': self._embed(["Put the white cube into the blue box"])[0].numpy(),
                }
                steps_data.append(step)

            print(f"Generated episode {episode_id} with {len(steps_data)} steps.")
            yield episode_id, {
                        'steps': steps_data,
                        'episode_metadata': {
                            # 这里与 _info 中的定义对应
                            'file_path': os.path.join(folder_path, npy_name),
                        }
                    }

