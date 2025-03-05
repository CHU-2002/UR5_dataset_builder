import tensorflow as tf
import tensorflow_datasets as tfds
import time

def main():
    ds, info = tfds.load("ur5_robo_dataset", split="train", with_info=True)
    print("Dataset info:\n", info)

    # Take one episode (one example)
    for example_idx, example in enumerate(ds.take(10)):
        print(f"\n[Example {example_idx}] has keys:", list(example.keys()))
        
        # example["steps"] is a Dataset (<_VariantDataset ...)
        steps_ds = example["steps"]  # This is a tf.data.Dataset, not a dictionary/tensor directly
        
        print("\nNow let's iterate over `steps` sub-dataset to inspect the `action`:")
        # Iterate over the sub-dataset of steps
        for step_idx, step_data in enumerate(steps_ds.take(300)):  # Here we only take the first 5 steps for example
            action_tensor = step_data["action"]
            # Print the shape and content of the tensor
            print(action_tensor.shape, action_tensor.numpy())
        
        # If you iterate over all steps without restriction, the data量可能非常大，容易导致内存溢出或等待很久
        # Therefore, here we only take the first 5 steps for example
        time.sleep(5)

if __name__ == "__main__":
    main()
