import pyrealsense2 as rs
import cv2
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import time
import asyncio
import numpy as np
import os
import threading
from vacuum_gripper import VacuumGripper

# Example camera serial dictionary
REALSENSE_CAMERAS = {
    "side_1": "218622277783",#405
    "side_2": "819612070593",#435
    "wrist_1": "130322272869" #D405
    # "side_2": "239722072823"
}


class CameraManager:
    """
    Responsibilities:
      1) Initialize multiple RealSense cameras and fetch images
      2) Collect color/depth images at a fixed frequency (e.g., ~5Hz) for visualization
      3) Use a custom RealTimeUR5Controller to get robot end-effector pose and joint positions
      4) Save images and robot data to local disk

    Example usage:
        camera_map = {
            "front": "123456789012",  # camera name -> serial
            "side":  "987654321098",
        }
        manager = CameraManager(
            robot_ip="192.168.1.60",
            camera_map=camera_map,
            save_path="data"
        )
        manager.init_cameras()
        manager.run()
    """

    def __init__(self, camera_map: dict, save_path,
                 UR5_controller=None, gripper_controller=None):
        """
        :param camera_map:     dict {camera_name: RealSense device serial}
        :param save_path:      Directory for saving data
        :param UR5_controller: Instance of RealTimeUR5Controller (optional)
        :param gripper_controller: Instance of AsyncGripperController or similar (optional)
        """
        self.camera_map = camera_map
        self.save_path = save_path

        self.pipelines = {}   # Holds camera_name -> rs.pipeline()
        self.index = 0        # Data saving counter

        # Example usage: RealTimeUR5Controller to get robot pose & joints
        self.ur5_controller = UR5_controller
        self.gripper_controller = gripper_controller

    def init_cameras(self):
        """
        Initialize the RealSense camera pipelines.
        """
        os.makedirs(self.save_path, exist_ok=True)

        for name, serial in self.camera_map.items():
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                pipeline.start(config)
                self.pipelines[name] = pipeline
                print(f"Camera [{name}] with serial={serial} started successfully.")
            except Exception as e:
                print(f"Error while starting camera [{name}] serial={serial}: {e}")
                self.pipelines[name] = None

    def run(self):
        """
        Main loop:
        Continuously read frames from cameras, retrieve robot data, visualize & save them.
        Press 'q' to exit.
        """
        try:
            while True:
                # Collect data from all cameras
                all_imgs = []
                for name, pipeline in self.pipelines.items():
                    if pipeline is None:
                        continue

                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                    if not color_frame or not depth_frame:
                        continue

                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())

                    # For demonstration: only store the first camera's color/depth
                    
                    all_imgs.append(color_image)
                    all_imgs.append(depth_image)

                    # Show color image in a window
                    cv2.imshow(f"Camera {name}", color_image)

                # Retrieve current robot pose & joints
                if self.ur5_controller:
                    pos, joint = self.ur5_controller.get_robot_pose_and_joints()
                else:
                    pos = [0.3, -0.2, 0.2, 0, 0, 0]  # default if no UR5 controller
                    joint = [0.0] * 6
                # Retrieve gripper state (could be 0/1 or other logic)
                if self.gripper_controller:
                    gripper_state = self.gripper_controller.get_gripper_state()
                else:
                    gripper_state = 0  # default if no gripper controller

                # If we have at least one camera's color+depth, save the data
                if len(all_imgs) >= 2:
                    self._save_data(
                        index=self.index,
                        joint=joint,
                        pos=pos,
                        imgs=all_imgs,
                        gripper=gripper_state
                    )
                    print(f"[INFO] Saved data at index={self.index}")
                    self.index += 1

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Exiting camera loop.")
                    break

                # ~5Hz => 0.2s
                time.sleep(0.2)

        except Exception as e:
            print(f"[ERROR] Camera streaming loop exception: {e}")
        finally:
            self.stop()

    def _save_data(self, index, joint, pos, imgs, gripper):
        """
        Save the current frame data to disk, including color/depth images and robot info.
        imgs: [primary_color_image, primary_depth_image,
            secondary_color_image, secondary_depth_image,
            wrist_color_image, wrist_depth_image]
        """
        # Initialize placeholders for images
        primary_color_img = primary_depth_img = None
        secondary_color_img = secondary_depth_img = None
        wrist_color_img = wrist_depth_img = None

        # Assign images based on length of imgs
        if len(imgs) >= 2:
            primary_color_img = imgs[0]
            primary_depth_img = imgs[1]
        if len(imgs) >= 4:
            secondary_color_img = imgs[2]
            secondary_depth_img = imgs[3]
        if len(imgs) >= 6:
            wrist_color_img = imgs[4]
            wrist_depth_img = imgs[5]

        # Pack into a dict
        data = {
            'joint': np.array(joint, dtype=np.float32),
            'pose':  np.array(pos,   dtype=np.float32),
            'primary_image': primary_color_img,
            'primary_depth_image': primary_depth_img,
            'secondary_image': secondary_color_img,
            'secondary_depth_image': secondary_depth_img,
            'wrist_image': wrist_color_img,
            'wrist_depth_image': wrist_depth_img,
            'gripper': gripper
        }

        # Save images for each camera if available
        if primary_color_img is not None:
            primary_color_path = os.path.join(self.save_path, f"primary_{index}.jpg")
            primary_depth_path = os.path.join(self.save_path, f"primary_depth_{index}.jpg")
            cv2.imwrite(primary_color_path, primary_color_img)
            if DEBUG:
                cv2.imwrite(primary_depth_path, primary_depth_img)

        if secondary_color_img is not None:
            secondary_color_path = os.path.join(self.save_path, f"secondary_{index}.jpg")
            secondary_depth_path = os.path.join(self.save_path, f"secondary_depth_{index}.jpg")
            if DEBUG:
                cv2.imwrite(secondary_color_path, secondary_color_img)
                cv2.imwrite(secondary_depth_path, secondary_depth_img)

        if wrist_color_img is not None:
            wrist_color_path = os.path.join(self.save_path, f"wrist_{index}.jpg")
            wrist_depth_path = os.path.join(self.save_path, f"wrist_depth_{index}.jpg")
            if DEBUG:
                cv2.imwrite(wrist_color_path, wrist_color_img)
                cv2.imwrite(wrist_depth_path, wrist_depth_img)

        # Save as .npy
        np.save(os.path.join(self.save_path, f"targ{index}.npy"), data)

    def stop(self):
        """
        Stop all camera streams, close OpenCV windows,
        and close the UR5 controller connection if needed.
        """
        for name, pipeline in self.pipelines.items():
            if pipeline is not None:
                pipeline.stop()
        cv2.destroyAllWindows()

        print("[INFO] CameraManager stopped.")

class RealTimeUR5Controller:
    """
    This sample class demonstrates how to update the UR5 end-effector pose in real-time
    (e.g., at 5Hz) using servoL, and also how to control the gripper or retrieve the
    robot's current pose/joint positions.

    Example usage:
        controller = RealTimeUR5Controller("192.168.0.100")
        controller.start_realtime_control(frequency=5.0)

        # In an external loop, keep updating the pose
        for step in range(50):
            new_pose = [0.3, -0.2 + 0.001*step, 0.2, 0, 0, 0]
            controller.send_pose_to_robot(new_pose, speed=1.0, acceleration=0.8)
            time.sleep(0.2)  # corresponding to 5Hz

        controller.stop_realtime_control()
        # Optionally control the gripper
        controller.control_gripper(close=True)
        # Also query current pose
        pose, joints = controller.get_robot_pose_and_joints()
        controller.close()
    """

    def __init__(self, ip_address: str, speed=0.1, acceleration=0.1):
        """
        :param ip_address: UR5 robot controller IP address
        """
        if ip_address:
            self.ip_address = ip_address
        else :
            self.ip_address = "192.168.1.60"

        # RTDE control & receive interfaces
        self.rtde_control = RTDEControlInterface(ip_address)
        self.rtde_receive = RTDEReceiveInterface(ip_address)

        self._running = False
        self._control_thread = None

        # Default: get the current pose from the robot
        self._target_pose = self.rtde_receive.getActualTCPPose()
        # Default speed & acceleration for moveL
        self._speed = speed
        self._acceleration = acceleration

        print(f"[INFO] Connected to UR5 at {ip_address}")

    def start_realtime_control(self, frequency=100.0):
        """
        Start a background thread that calls servoL() at the specified frequency (Hz),
        so the robot EEF follows self._target_pose.
        """
        if self._running:
            print("[WARNING] Real-time control is already running.")
            return

        self._running = True
        self._control_thread = threading.Thread(
            target=self._control_loop,
            args=(frequency,),
            daemon=True
        )
        self._control_thread.start()
        print(f"[INFO] Real-time control started at {frequency} Hz.")

    def _control_loop(self, frequency):
        """
        Background thread: call servoL() every 1/frequency seconds, moving toward self._target_pose.
        """
        period = 1.0 / frequency
        while self._running:
            self.rtde_control.servoL(
                self._target_pose,
                self._speed,
                self._acceleration,
                period,
                0.1,
                300
            )
            time.sleep(period)

        # If you want to stop the motion gracefully, call stopL()
        self.rtde_control.stopL(2.0)
        print("[INFO] Real-time control loop stopped.")

    def stop_realtime_control(self):
        """
        Stop the background real-time control thread.
        """
        if not self._running:
            return
        self._running = False
        if self._control_thread is not None:
            self._control_thread.join()
        print("[INFO] Real-time control stopped.")

    def send_pose_to_robot(self, pose, speed=0.1, acceleration=0.1):
        """
        External call to update self._target_pose.
        The background thread will keep calling servoL() to move toward this new pose.

        :param pose: [x, y, z, Rx, Ry, Rz] (Rx, Ry, Rz as axis-angle)
        :param speed: (m/s)
        :param acceleration: (m/s^2)
        """
        self._target_pose = pose
        self._speed = speed
        self._acceleration = acceleration

    def get_robot_pose_and_joints(self):
        """
        Get the current EEF pose (x, y, z, Rx, Ry, Rz) and 6 joint angles.

        Returns: (task_space_pose, joint_space_positions)
        """
        try:
            # Current TCPPose
            task_space_pose = self.rtde_receive.getActualTCPPose()
            # Example: negate Rx, Ry, Rz
            task_space_pose = task_space_pose[:3] + [-val for val in task_space_pose[3:]]

            # Current joint angles
            joint_space_positions = self.rtde_receive.getActualQ()
            return task_space_pose, joint_space_positions
        except Exception as e:
            print("Error while retrieving robot data:", e)
            return None, None

    def close(self):
        """
        Stop the background control thread and release RTDE resources.
        """
        self.stop_realtime_control()
        self.rtde_control.stopScript()
        print("[INFO] Robot controller closed.")


class AsyncGripperController:
    """
    This class demonstrates how to establish a single VacuumGripper connection and
    reuse it throughout the class lifecycle for repeated open/close operations
    without reconnecting each time.

    Example usage:
        controller = AsyncGripperController("192.168.0.100")
        # Open the gripper
        controller.control_gripper(close=False, force=80, speed=25)
        # Close the gripper
        controller.control_gripper(close=True, force=100, speed=30)
        # Finally disconnect
        controller.disconnect()
    """

    def __init__(self, ip_address: str):
        self.ip_address = ip_address
        self.loop = None
        self.gripper = None
        self._connected = False
        self._gripper_state = False  # False for open, True for closed

    def _ensure_event_loop(self):
        """
        Ensure self.loop exists and is usable. Create a new loop if none is found.
        """
        if self.loop is None:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)

    async def _init_gripper(self):
        """
        Asynchronously initialize and activate the vacuum gripper, only once.
        """
        if self.gripper is None:
            self.gripper = VacuumGripper(self.ip_address)

        if not self._connected:
            await self.gripper.connect()
            await self.gripper.activate()
            self._connected = True

    def get_gripper_state(self):
        return self._gripper_state

    def _run_async_task(self, coro):
        """
        Run the coroutine in an existing (or newly created) event loop.
        If the loop is already running, use ensure_future; otherwise run_until_complete.
        """
        self._ensure_event_loop()

        if self.loop.is_running():
            asyncio.ensure_future(coro, loop=self.loop)
        else:
            self.loop.run_until_complete(coro)

    def control_gripper(self, close=True, force=100, speed=30):
        """
        Control gripper open/close:
         - close=True for closing
         - close=False for opening
         - force & speed for gripper settings
        """
        async def _control():
            await self._init_gripper()
            if close:
                self._gripper_state = True
                await self.gripper.close_gripper(force=force, speed=speed)
                print("Gripper closed.")
            else:
                self._gripper_state = False
                await self.gripper.open_gripper(force=force, speed=speed)
                print("Gripper opened.")

        self._run_async_task(_control())

    def disconnect(self):
        """
        Disconnect from the gripper. The next control_gripper() call will reconnect.
        """
        async def _disconnect():
            if self.gripper and self._connected:
                await self.gripper.disconnect()
                self._connected = False
                self.gripper = None

        self._run_async_task(_disconnect())

    def __del__(self):
        """
        Destructor: attempt to disconnect on garbage collection.
        It's recommended to explicitly call disconnect() rather than rely on __del__.
        """
        try:
            self.disconnect()
        except:
            pass


def main():
    UR5_IP = "192.168.1.60"
    SAVE_PATH = "data"

    # 1. Instantiate UR5 and gripper controllers
    ur5_controller = RealTimeUR5Controller(UR5_IP)
    gripper_controller = AsyncGripperController(UR5_IP)

    # 2. Instantiate CameraManager

    # 3. Initialize cameras
    camera_manager.init_cameras()
    # 4. Prepare a thread to run camera_manager.run() for collecting and storing data
    camera_thread = threading.Thread(target=camera_manager.run, daemon=True)
    # camera_thread.start()
    # 5. Start a simple SpaceMouse control
    spacemouse = SpaceMouseExpert()  # You need to implement or use a third-party library
    camera_running = False
    zero_count = 0
    ur5_controller.start_realtime_control(frequency=5.0)
    # Open the gripper initially
    gripper_controller.control_gripper(close=False, force=100, speed=30)

    # Set an initial robot pose
    current_pose = [-0.470, -0.180, 0.450, np.pi, 0, 0]
    print("current_pose: ", current_pose)
    print("type",type(current_pose))
    
    ur5_controller.send_pose_to_robot(current_pose)

    print("[INFO] Start main loop for SpaceMouse + Robot ...")

    try:
        while True:
            # Get SpaceMouse movement (action) and buttons
            action, buttons = spacemouse.get_action()
            print(f"Spacemouse action: {action}, buttons: {buttons}")

            # Update current_pose
            for i in range(6):
                current_pose[i] += action[i] * 0.05

            # Send new EEF pose
            ur5_controller.send_pose_to_robot(current_pose, speed=0.8, acceleration=0.8)

            # Use button[0] to control gripper
            close_gripper = bool(buttons[0])
            gripper_controller.control_gripper(close=close_gripper, force=100, speed=100)

            # Check if there's any movement/button input
            if any(abs(a) > 1e-5 for a in action) or any(buttons):
                zero_count = 0
                if not camera_running:
                    # Start camera thread
                    camera_thread.start()
                    camera_running = True
                    print("Camera thread started.")
            # else:
            #     zero_count += 1
            #     # If no input for 5 cycles, exit
            #     if zero_count >= 5:
            #         if camera_running:
            #             print("Stopping main loop since no input for 5 cycles.")
            #         break

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt: stopping main loop.")
    finally:
        # 6. Stop CameraManager
        camera_manager.stop()
        if camera_thread.is_alive():
            camera_thread.join()

        # 7. Disconnect gripper and close UR5 RTDE
        gripper_controller.disconnect()
        ur5_controller.close()

        print("[INFO] Main program finished.")
    # time.sleep(10)
    # camera_manager.stop()
    # # if camera_thread.is_alive():
    # #     camera_thread.join()


if __name__ == "__main__":
    main()
