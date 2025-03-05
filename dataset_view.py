import tensorflow as tf
import tensorflow_datasets as tfds
import time

def main():
    ds, info = tfds.load("ur5_robo_dataset", split="train", with_info=True)
    print("Dataset info:\n", info)

    # 只取一条 episode （也就是一条 example）
    for example_idx, example in enumerate(ds.take(10)):
        print(f"\n[Example {example_idx}] has keys:", list(example.keys()))
        
        # example["steps"] 是一个 Dataset (<_VariantDataset ...>)
        steps_ds = example["steps"]  # 这是一个 tf.data.Dataset，而非直接的字典/张量
        
        print("\nNow let's iterate over `steps` sub-dataset to inspect the `action`:")
        # 遍历子 Dataset 里的 step
        for step_idx, step_data in enumerate(steps_ds.take(300)):  # 这里只取前5个 step 做示例
            action_tensor = step_data["action"]
            # 打印张量的形状、内容
            print(action_tensor.shape, action_tensor.numpy())
        
        # 如不加限制地遍历所有 step，数据量可能非常大，容易导致内存溢出或等待很久
        # 因此这里示例仅取前5个 step
        time.sleep(5)

if __name__ == "__main__":
    main()
