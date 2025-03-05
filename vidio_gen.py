import numpy as np
import cv2
import os
import time

def load_and_save_video(data_folder="data", output_video_path="output.mp4"):
    """
    Read the targ{i}.npy file in data_folder,
    print the pose / gripper information, and write the primary_image to the video file.
    """
    # Define video encoder and output file
    fps = 20  # You can adjust the frame rate as needed
    frame_size = (640, 480)  # Uniform video frame size
    
    # Use mp4v encoder, ensure compatibility
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    i = 0  # Start from the 0th file
    while True:
        # Construct the file name
        file_path = os.path.join(data_folder, f"targ{i}.npy")

        # If the file does not exist, stop
        if not os.path.exists(file_path):
            time.sleep(5)
            print(f"[INFO] The file {file_path} does not exist, reading ends.")
            break

        # Read the .npy file
        data = np.load(file_path, allow_pickle=True).item()

        # Get the primary_image, pose, gripper information
        primary_image = data.get('primary_image', None)
        pose = data.get('pose', None)
        gripper_val = data.get('gripper', None)

        # Print the key information
        print(f"\nCurrent index: {i}")
        if pose is not None:
            print("Pose data:", pose)  # Like [x, y, z, Rx, Ry, Rz]
        else:
            print("[WARNING] The file does not contain 'pose' data.")

        if gripper_val is not None:
            print("Gripper data:", gripper_val)
        else:
            print("[WARNING] The file does not contain 'gripper' data.")

        # Process the primary_image
        if primary_image is not None:
            print(f"[DEBUG] The original primary_image shape: {primary_image.shape}, dtype: {primary_image.dtype}")

            # Ensure the primary_image data type is correct
            if primary_image.dtype != np.uint8:
                primary_image = (primary_image * 255).astype(np.uint8)

            # Ensure the primary_image is 3 channels
            if len(primary_image.shape) == 2:  # Grayscale image
                primary_image = cv2.cvtColor(primary_image, cv2.COLOR_GRAY2BGR)

            elif primary_image.shape[2] == 4:  # Maybe RGBA, need to convert to BGR
                primary_image = cv2.cvtColor(primary_image, cv2.COLOR_RGBA2BGR)

            # Resize the image to (336, 336)
            primary_image_resized = cv2.resize(primary_image, frame_size, interpolation=cv2.INTER_LINEAR)

            # OpenCV needs BGR format, so no need to convert RGB
            print(f"[DEBUG] The processed primary_image shape: {primary_image_resized.shape}, dtype: {primary_image_resized.dtype}")

            # Write the image to the video
            out.write(primary_image_resized)

        else:
            print("[WARNING] The file does not contain 'primary_image' data.")

        i += 1
        time.sleep(0.05)

    # Release the video writer object
    out.release()
    print(f"[INFO] The video has been saved to: {output_video_path}")

if __name__ == "__main__":
    # Run the code
    load_and_save_video(data_folder="results/task14", output_video_path="output.mp4")
