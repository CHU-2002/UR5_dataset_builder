import os
import numpy as np
import shutil
from glob import glob

def load_dataset(data_dir):
    npy_files = glob(os.path.join(data_dir, "targ*.npy"))
    
    steps = []
    for f in npy_files:
        basename = os.path.basename(f)
        index_str = ''.join([c for c in basename if c.isdigit()])
        if index_str == '':
            continue
        index = int(index_str)
        data = np.load(f, allow_pickle=True).item()
        pose = data["pose"]  
        steps.append({'filename': basename, 'fullpath': f, 'index': index, 'pose': pose, 'data': data})
    
    steps.sort(key=lambda x: x['index'])
    return steps

def filter_same_pose(steps, eps=1e-9):
    filtered = []
    last_pose = None
    
    for i, s in enumerate(steps):
        pose = s['pose']
        if last_pose is None or np.linalg.norm(pose - last_pose) > eps:
            filtered.append(s)
            last_pose = pose
    
    if steps and steps[-1] not in filtered:
        filtered.append(steps[-1])  # Ensure the last frame is retained
    
    return filtered

def filter_small_displacement(steps):
    if len(steps) <= 1:
        return steps
    
    displacements = [np.linalg.norm(steps[i+1]['pose'][:3] - steps[i]['pose'][:3]) for i in range(len(steps) - 1)]
    avg_disp = np.mean(displacements)
    half_avg = avg_disp / 2.0
    
    keep_flags = [True] * len(steps)
    for i, d in enumerate(displacements):
        if d < half_avg:
            keep_flags[i+1] = False
    
    filtered = [step for step, keep in zip(steps, keep_flags) if keep]
    if steps[-1] not in filtered:
        filtered.append(steps[-1])  # Ensure the last frame exists
    
    return filtered

def save_filtered_steps(filtered_steps, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    for new_index, s in enumerate(filtered_steps):
        data_dict = s['data']
        new_filename = f"targ{new_index}.npy"
        dst_path = os.path.join(save_dir, new_filename)
        np.save(dst_path, data_dict)

def process_one_task(original_data_dir, filtered_data_dir, eps=1e-9):
    steps = load_dataset(original_data_dir)
    if len(steps) == 0:
        print(f"[WARN] The directory {original_data_dir} does not contain .npy data, skipping.")
        return
    
    print(f"Task dir: {original_data_dir}, total steps: {len(steps)}")
    
    steps_no_dup = filter_same_pose(steps, eps=eps)
    steps_final = filter_small_displacement(steps_no_dup)
    
    save_filtered_steps(steps_final, filtered_data_dir)
    
    print(f"  After removing duplicates: {len(steps_no_dup)}")
    print(f"  After removing small disp : {len(steps_final)}")
    print(f"  => saved to {filtered_data_dir}")

def main():
    for task_id in range(1, 101):
        original_data_dir = f"data/task{task_id}"
        filtered_data_dir = f"data_clean/task{task_id}"
        process_one_task(original_data_dir, filtered_data_dir, eps=1e-9)

if __name__ == "__main__":
    main()
