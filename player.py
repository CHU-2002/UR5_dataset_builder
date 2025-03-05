import numpy as np
import cv2
import os
import time

def load_and_show_data(data_folder="data"):
    """
    Read the targ{i}.npy files in data_folder one by one,
    print the pose and display the primary_image.
    Press 'q' to exit the loop.
    """
    i = 0  # Start from the 0th file
    while True:
        # Construct the file name, e.g. data/targ0.npy, data/targ1.npy, etc.
        file_path = os.path.join(data_folder, f"targ{i}.npy")
        # If the file does not exist, stop
        if not os.path.exists(file_path):
            time.sleep(5)
            print(f"[INFO] The file {file_path} does not exist, reading ends.")
            break

        # Read the .npy file
        data = np.load(file_path, allow_pickle=True).item()  # Use .item() to get the dictionary

        # Get the primary_image and pose
        primary_image = data.get('primary_image', None)
        primary_image = primary_image[..., ::-1]  # BGR->RGB
        #  primary_image = primary_image[..., ::-1]  # BGR->RGB
        primary_image = cv2.resize(primary_image, (336, 336), interpolation=cv2.INTER_LINEAR)
        pose = data.get('pose', None)
        gripper_val = data.get('gripper', None)
        secondary_image = data.get('secondary_image', None)
        wrist_image = data.get('wrist_image', None)
        dep_primary_image = data.get('primary_depth_image', None)

        # Print the key information
        print(f"Current index: {i}")
        if pose is not None:
            print("Pose data:", pose)  # Like [x, y, z, Rx, Ry, Rz]
        else:
            print("[WARNING] The file does not contain 'pose' data.")
        if gripper_val is not None:
            print("Gripper data:", gripper_val)
        else:
            print("[WARNING] The file does not contain 'gripper' data.")
        if gripper_val > 0.5:
            time.sleep(0.2)
        # Display the primary_image
        if wrist_image is not None:
            cv2.imshow("Primary Image", primary_image)
        else:

            print("[WARNING] The file does not contain 'primary_image' data.")

        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting camera loop.")
            break
        time.sleep(0.1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example: Specify the folder to save the data
    load_and_show_data(data_folder="data_clean/task33")
