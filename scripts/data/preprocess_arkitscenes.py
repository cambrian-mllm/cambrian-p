import os
import json
import os.path as osp
import decimal
import argparse
import math
from bisect import bisect_left
from PIL import Image
import numpy as np
import quaternion
from scipy import interpolate
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arkitscenes_dir",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker threads. Defaults to CPU count."
    )
    return parser


def value_to_decimal(value, decimal_places):
    decimal.getcontext().rounding = decimal.ROUND_HALF_UP  # define rounding method
    return decimal.Decimal(str(float(value))).quantize(
        decimal.Decimal("1e-{}".format(decimal_places))
    )


def closest(value, sorted_list):
    index = bisect_left(sorted_list, value)
    if index == 0:
        return sorted_list[0]
    elif index == len(sorted_list):
        return sorted_list[-1]
    else:
        value_before = sorted_list[index - 1]
        value_after = sorted_list[index]
        if value_after - value < value - value_before:
            return value_after
        else:
            return value_before


def get_up_vectors(pose_device_to_world):
    return np.matmul(pose_device_to_world, np.array([[0.0], [-1.0], [0.0], [0.0]]))


def get_right_vectors(pose_device_to_world):
    return np.matmul(pose_device_to_world, np.array([[1.0], [0.0], [0.0], [0.0]]))


def read_traj(traj_path):
    quaternions = []
    poses = []
    timestamps = []
    poses_p_to_w = []
    with open(traj_path) as f:
        traj_lines = f.readlines()
        for line in traj_lines:
            tokens = line.split()
            assert len(tokens) == 7
            traj_timestamp = float(tokens[0])

            timestamps_decimal_value = value_to_decimal(traj_timestamp, 3)
            timestamps.append(
                float(timestamps_decimal_value)
            )  # for spline interpolation

            angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            r_w_to_p, _ = cv2.Rodrigues(np.asarray(angle_axis))
            t_w_to_p = np.asarray(
                [float(tokens[4]), float(tokens[5]), float(tokens[6])]
            )

            pose_w_to_p = np.eye(4)
            pose_w_to_p[:3, :3] = r_w_to_p
            pose_w_to_p[:3, 3] = t_w_to_p

            pose_p_to_w = np.linalg.inv(pose_w_to_p)

            r_p_to_w_as_quat = quaternion.from_rotation_matrix(pose_p_to_w[:3, :3])
            t_p_to_w = pose_p_to_w[:3, 3]
            poses_p_to_w.append(pose_p_to_w)
            poses.append(t_p_to_w)
            quaternions.append(r_p_to_w_as_quat)
    return timestamps, poses, quaternions, poses_p_to_w


def get_available_frames(rgb_dir, depth_dir):
    """Get list of available frames that have both RGB and depth."""
    if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir):
        return []

    rgb_frames = set([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    depth_frames = set([f for f in os.listdir(depth_dir) if f.endswith('.png')])

    # Get intersection of available frames
    available_frames = sorted(list(rgb_frames.intersection(depth_frames)))
    return available_frames


def uniform_sample_frames(available_frames, num_frames):
    """Uniformly sample num_frames from available frames."""
    if len(available_frames) <= num_frames:
        return available_frames

    # Sort frames by their numeric timestamp
    sorted_frames = sorted(available_frames, key=lambda x: float(x.split('_')[1].split('.png')[0]))

    # Uniform sampling
    indices = np.linspace(0, len(sorted_frames) - 1, num_frames, dtype=int)
    sampled_frames = [sorted_frames[i] for i in indices]

    return sampled_frames


def process_scene_data(scene_subdir, scene_dir, timestamps, poses, quaternions, poses_cam_to_world,
                      selected_images, intrinsics_dir):
    """Process scene data and return metadata."""
    timestamps_selected = [float(frame_id) for _, frame_id in selected_images]

    sky_direction_scene, trajectories, intrinsics, images = (
        convert_scene_metadata(
            scene_subdir,
            intrinsics_dir,
            timestamps,
            quaternions,
            poses,
            poses_cam_to_world,
            selected_images,
            timestamps_selected,
        )
    )

    if len(images) == 0:
        return None

    # Create pairs - all possible pairs from sampled frames
    num_selected = len(images)
    pairs = []
    for i in range(num_selected):
        for j in range(i + 1, num_selected):
            pairs.append([i, j])
    pairs = np.array(pairs) if pairs else np.empty((0, 2), dtype=int)

    return {
        'sky_direction_scene': sky_direction_scene,
        'trajectories': trajectories,
        'intrinsics': intrinsics,
        'images': images,
        'pairs': pairs
    }


def save_single_image(args):
    """Save a single converted image (RGB or depth). Used for parallel processing."""
    basename, vga_wide_path, depth_path, img_out, depth_out, sky_direction_scene, save_type = args

    if save_type == 'both':
        if osp.isfile(img_out) and osp.isfile(depth_out):
            return True
    elif save_type == 'rgb' and osp.isfile(img_out):
        return True
    elif save_type == 'depth' and osp.isfile(depth_out):
        return True

    try:
        if save_type in ['rgb', 'both']:
            img = Image.open(vga_wide_path)

            # rotate the image
            if sky_direction_scene == "RIGHT":
                try:
                    img = img.transpose(Image.Transpose.ROTATE_90)
                except Exception:
                    img = img.transpose(Image.ROTATE_90)
            elif sky_direction_scene == "LEFT":
                try:
                    img = img.transpose(Image.Transpose.ROTATE_270)
                except Exception:
                    img = img.transpose(Image.ROTATE_270)
            elif sky_direction_scene == "DOWN":
                try:
                    img = img.transpose(Image.Transpose.ROTATE_180)
                except Exception:
                    img = img.transpose(Image.ROTATE_180)

            if not osp.isfile(img_out):
                img.save(img_out)

            W, H = img.size

        if save_type in ['depth', 'both']:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            # rotate the depth
            if sky_direction_scene == "RIGHT":
                depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif sky_direction_scene == "LEFT":
                depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
            elif sky_direction_scene == "DOWN":
                depth = cv2.rotate(depth, cv2.ROTATE_180)

            if save_type == 'depth':
                # Need to get dimensions from RGB image
                img = Image.open(vga_wide_path)
                if sky_direction_scene in ["RIGHT", "LEFT"]:
                    W, H = img.size[1], img.size[0]  # swapped
                else:
                    W, H = img.size

            depth = cv2.resize(
                depth, (W, H), interpolation=cv2.INTER_NEAREST_EXACT
            )
            if not osp.isfile(depth_out):
                cv2.imwrite(depth_out, depth)

        return True
    except Exception as e:
        print(f"Error processing {basename}: {e}")
        return False


def save_converted_images_parallel(scene_subdir, out_scene_subdir, rgb_dir, depth_dir,
                                  sky_direction_scene, all_available_frames, num_workers=None):
    """Save converted images (RGB and depth) for ALL available frames using parallel processing."""
    os.makedirs(os.path.join(out_scene_subdir, "vga_wide"), exist_ok=True)
    os.makedirs(os.path.join(out_scene_subdir, "lowres_depth"), exist_ok=True)

    # Check if all images exist
    for basename in all_available_frames:
        vga_wide_path = osp.join(rgb_dir, basename)
        depth_path = osp.join(depth_dir, basename)
        if not osp.isfile(vga_wide_path) or not osp.isfile(depth_path):
            return False

    # Prepare arguments for parallel processing
    tasks = []
    for basename in all_available_frames:
        img_out = os.path.join(
            out_scene_subdir, "vga_wide", basename.replace(".png", ".jpg")
        )
        depth_out = os.path.join(out_scene_subdir, "lowres_depth", basename)
        vga_wide_path = osp.join(rgb_dir, basename)
        depth_path = osp.join(depth_dir, basename)

        tasks.append((
            basename, vga_wide_path, depth_path, img_out, depth_out,
            sky_direction_scene, 'both'
        ))

    # Process images in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(save_single_image, tasks))

    return all(results)


def process_single_scene(args):
    """Process a single scene. This function is designed to be run in parallel."""
    scene_subdir, rootdir, subdir, outsubdir = args

    out_scene_subdir = osp.join(outsubdir, scene_subdir)
    os.makedirs(out_scene_subdir, exist_ok=True)

    scene_dir = osp.join(rootdir, subdir, scene_subdir)
    depth_dir = osp.join(scene_dir, "lowres_depth")
    rgb_dir = osp.join(scene_dir, "vga_wide")
    intrinsics_dir = osp.join(scene_dir, "vga_wide_intrinsics")
    traj_path = osp.join(scene_dir, "lowres_wide.traj")

    # Check if required directories exist
    if not all(os.path.exists(p) for p in [depth_dir, rgb_dir, intrinsics_dir, traj_path]):
        return None

    # Get available frames
    available_frames = get_available_frames(rgb_dir, depth_dir)
    if len(available_frames) == 0:
        return None

    # Load trajectory data once
    timestamps, poses, quaternions, poses_cam_to_world = read_traj(traj_path)
    poses = np.array(poses)
    quaternions = np.array(quaternions, dtype=np.quaternion)
    quaternions = quaternion.unflip_rotors(quaternions)
    timestamps = np.array(timestamps)

    # Process different sampling strategies
    sampling_configs = [
        ('uni32', 32),
        ('uni128', 128),
        ('all', None)  # None means use all frames
    ]

    scene_processed = False
    sky_direction_scene = None

    for suffix, num_frames in sampling_configs:
        scene_metadata_path = osp.join(out_scene_subdir, f"scene_metadata_{suffix}.npz")

        if osp.isfile(scene_metadata_path):
            scene_processed = True
            continue

        # Get selection based on sampling strategy
        if num_frames is None:
            # Use all available frames
            selection = sorted(available_frames, key=lambda x: float(x.split('_')[1].split('.png')[0]))
            print(f"parsing {scene_subdir} - ALL {len(selection)} frames from {len(available_frames)} available")
        else:
            # Uniform sampling
            selection = uniform_sample_frames(available_frames, num_frames)
            print(f"parsing {scene_subdir} - {len(selection)} frames selected from {len(available_frames)} available (uni{num_frames})")

        if len(selection) == 0:
            continue

        selected_images = [
            (basename, basename.split(".png")[0].split("_")[1])
            for basename in selection
        ]

        # Process scene data
        scene_data = process_scene_data(
            scene_subdir, scene_dir, timestamps, poses, quaternions,
            poses_cam_to_world, selected_images, intrinsics_dir
        )

        if scene_data is None:
            print(f"Warning: No valid frames found for scene {scene_subdir} with {suffix}, skipping")
            continue

        # Store sky direction from first successful processing
        if sky_direction_scene is None:
            sky_direction_scene = scene_data['sky_direction_scene']

        # Save converted images (only need to do this once)
        if not scene_processed:
            # Use parallel processing for image conversion
            success = save_converted_images_parallel(
                scene_subdir, out_scene_subdir, rgb_dir, depth_dir,
                sky_direction_scene, available_frames, num_workers=4  # Use fewer threads per scene
            )
            if not success:
                continue
            scene_processed = True

        # Save scene metadata
        np.savez(
            scene_metadata_path,
            trajectories=scene_data['trajectories'],
            intrinsics=scene_data['intrinsics'],
            images=scene_data['images'],
            pairs=scene_data['pairs'],
        )

    if scene_processed:
        return scene_subdir
    return None


def main(rootdir, outdir, num_workers=None):
    os.makedirs(outdir, exist_ok=True)

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    print(f"Using {num_workers} worker threads")

    subdirs = ["Test", "Training"]
    for subdir in subdirs:
        # STEP 1: list all scenes
        outsubdir = osp.join(outdir, subdir)
        os.makedirs(outsubdir, exist_ok=True)

        # Find all scene directories
        subdir_path = osp.join(rootdir, subdir)
        if not os.path.exists(subdir_path):
            continue

        scene_dirs = [d for d in os.listdir(subdir_path)
                     if os.path.isdir(osp.join(subdir_path, d))]

        # Prepare arguments for parallel processing
        scene_args = [(scene_subdir, rootdir, subdir, outsubdir)
                      for scene_subdir in scene_dirs]

        # Process scenes in parallel
        valid_scenes = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_scene = {
                executor.submit(process_single_scene, args): args[0]
                for args in scene_args
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(scene_args), desc=f"Processing {subdir}") as pbar:
                for future in as_completed(future_to_scene):
                    scene_subdir = future_to_scene[future]
                    try:
                        result = future.result()
                        if result is not None:
                            valid_scenes.append(result)
                    except Exception as exc:
                        print(f"Scene {scene_subdir} generated an exception: {exc}")
                    pbar.update(1)

        # Save scene lists and create combined metadata for each sampling strategy
        for suffix, num_frames in [('uni32', 32), ('uni128', 128), ('all', None)]:
            # Filter valid scenes for this specific sampling strategy
            valid_scenes_for_suffix = []
            for scene_subdir in valid_scenes:
                scene_metadata_path = osp.join(
                    outsubdir, scene_subdir, f"scene_metadata_{suffix}.npz"
                )
                if osp.isfile(scene_metadata_path):
                    valid_scenes_for_suffix.append(scene_subdir)

            # Save scene list
            outlistfile = osp.join(outsubdir, f"scene_list_{suffix}.json")
            with open(outlistfile, "w") as f:
                json.dump(valid_scenes_for_suffix, f)

            # STEP 5: concat all scene_metadata.npz into a single file
            if not valid_scenes_for_suffix:
                continue

            scene_data = {}
            for scene_subdir in valid_scenes_for_suffix:
                scene_metadata_path = osp.join(
                    outsubdir, scene_subdir, f"scene_metadata_{suffix}.npz"
                )
                with np.load(scene_metadata_path) as data:
                    trajectories = data["trajectories"]
                    intrinsics = data["intrinsics"]
                    images = data["images"]
                    pairs = data["pairs"]
                scene_data[scene_subdir] = {
                    "trajectories": trajectories,
                    "intrinsics": intrinsics,
                    "images": images,
                    "pairs": pairs,
                }

            offset = 0
            counts = []
            scenes = []
            sceneids = []
            images = []
            intrinsics = []
            trajectories = []
            pairs = []

            for scene_idx, (scene_subdir, data) in enumerate(scene_data.items()):
                num_imgs = data["images"].shape[0]
                img_pairs = data["pairs"]

                scenes.append(scene_subdir)
                sceneids.extend([scene_idx] * num_imgs)

                images.append(data["images"])

                K = np.expand_dims(np.eye(3), 0).repeat(num_imgs, 0)
                K[:, 0, 0] = [fx for _, _, fx, _, _, _ in data["intrinsics"]]
                K[:, 1, 1] = [fy for _, _, _, fy, _, _ in data["intrinsics"]]
                K[:, 0, 2] = [hw for _, _, _, _, hw, _ in data["intrinsics"]]
                K[:, 1, 2] = [hh for _, _, _, _, _, hh in data["intrinsics"]]

                intrinsics.append(K)
                trajectories.append(data["trajectories"])

                # offset pairs
                if len(img_pairs) > 0:
                    img_pairs[:, 0:2] += offset
                pairs.append(img_pairs)
                counts.append(offset)

                offset += num_imgs

            images = np.concatenate(images, axis=0)
            intrinsics = np.concatenate(intrinsics, axis=0)
            trajectories = np.concatenate(trajectories, axis=0)
            pairs = np.concatenate(pairs, axis=0) if pairs and any(len(p) > 0 for p in pairs) else np.empty((0, 2), dtype=int)

            np.savez(
                osp.join(outsubdir, f"all_metadata_{suffix}.npz"),
                counts=counts,
                scenes=scenes,
                sceneids=sceneids,
                images=images,
                intrinsics=intrinsics,
                trajectories=trajectories,
                pairs=pairs,
            )


def convert_scene_metadata(
    scene_subdir,
    intrinsics_dir,
    timestamps,
    quaternions,
    poses,
    poses_cam_to_world,
    selected_images,
    timestamps_selected,
):
    # find scene orientation
    sky_direction_scene, rotated_to_cam = find_scene_orientation(poses_cam_to_world)

    # find/compute pose for selected timestamps
    # most images have a valid timestamp / exact pose associated
    timestamps_selected = np.array(timestamps_selected)

    # Filter out timestamps that are outside the trajectory range
    min_timestamp = timestamps.min()
    max_timestamp = timestamps.max()

    valid_indices = []
    valid_timestamps = []
    for i, ts in enumerate(timestamps_selected):
        if min_timestamp <= ts <= max_timestamp:
            valid_indices.append(i)
            valid_timestamps.append(ts)

    if len(valid_indices) == 0:
        print(f"Warning: No valid timestamps found for scene {scene_subdir}")
        return sky_direction_scene, [], [], []

    valid_timestamps = np.array(valid_timestamps)

    spline = interpolate.interp1d(timestamps, poses, kind="linear", axis=0)
    interpolated_rotations = quaternion.squad(
        quaternions, timestamps, valid_timestamps
    )
    interpolated_positions = spline(valid_timestamps)

    trajectories = []
    intrinsics = []
    images = []

    # Only process valid images
    valid_selected_images = [selected_images[i] for i in valid_indices]

    for idx, (i, (basename, frame_id)) in enumerate(zip(valid_indices, valid_selected_images)):
        intrinsic_fn = osp.join(intrinsics_dir, f"{scene_subdir}_{frame_id}.pincam")
        if not osp.exists(intrinsic_fn):
            intrinsic_fn = osp.join(
                intrinsics_dir, f"{scene_subdir}_{float(frame_id) - 0.001:.3f}.pincam"
            )
        if not osp.exists(intrinsic_fn):
            intrinsic_fn = osp.join(
                intrinsics_dir, f"{scene_subdir}_{float(frame_id) + 0.001:.3f}.pincam"
            )
        if not osp.exists(intrinsic_fn):
            print(f"Warning: Intrinsic file not found for frame {frame_id} in scene {scene_subdir}, skipping this frame")
            continue

        w, h, fx, fy, hw, hh = np.loadtxt(intrinsic_fn)  # PINHOLE

        pose = np.eye(4)
        pose[:3, :3] = quaternion.as_rotation_matrix(interpolated_rotations[idx])
        pose[:3, 3] = interpolated_positions[idx]

        images.append(basename)
        if sky_direction_scene == "RIGHT" or sky_direction_scene == "LEFT":
            intrinsics.append([h, w, fy, fx, hh, hw])  # swapped intrinsics
        else:
            intrinsics.append([w, h, fx, fy, hw, hh])
        trajectories.append(
            pose @ rotated_to_cam
        )  # pose_cam_to_world @ rotated_to_cam = rotated(cam) to world

    return sky_direction_scene, trajectories, intrinsics, images


def find_scene_orientation(poses_cam_to_world):
    if len(poses_cam_to_world) > 0:
        up_vector = sum(get_up_vectors(p) for p in poses_cam_to_world) / len(
            poses_cam_to_world
        )
        right_vector = sum(get_right_vectors(p) for p in poses_cam_to_world) / len(
            poses_cam_to_world
        )
        up_world = np.array([[0.0], [0.0], [1.0], [0.0]])
    else:
        up_vector = np.array([[0.0], [-1.0], [0.0], [0.0]])
        right_vector = np.array([[1.0], [0.0], [0.0], [0.0]])
        up_world = np.array([[0.0], [0.0], [1.0], [0.0]])

    # value between 0, 180
    device_up_to_world_up_angle = (
        np.arccos(np.clip(np.dot(np.transpose(up_world), up_vector), -1.0, 1.0)).item()
        * 180.0
        / np.pi
    )
    device_right_to_world_up_angle = (
        np.arccos(
            np.clip(np.dot(np.transpose(up_world), right_vector), -1.0, 1.0)
        ).item()
        * 180.0
        / np.pi
    )

    up_closest_to_90 = abs(device_up_to_world_up_angle - 90.0) < abs(
        device_right_to_world_up_angle - 90.0
    )
    if up_closest_to_90:
        assert abs(device_up_to_world_up_angle - 90.0) < 45.0
        # LEFT
        if device_right_to_world_up_angle > 90.0:
            sky_direction_scene = "LEFT"
            cam_to_rotated_q = quaternion.from_rotation_vector(
                [0.0, 0.0, math.pi / 2.0]
            )
        else:
            # note that in metadata.csv RIGHT does not exist, but again it's not accurate...
            # well, turns out there are scenes oriented like this
            # for example Training/41124801
            sky_direction_scene = "RIGHT"
            cam_to_rotated_q = quaternion.from_rotation_vector(
                [0.0, 0.0, -math.pi / 2.0]
            )
    else:
        # right is close to 90
        assert abs(device_right_to_world_up_angle - 90.0) < 45.0
        if device_up_to_world_up_angle > 90.0:
            sky_direction_scene = "DOWN"
            cam_to_rotated_q = quaternion.from_rotation_vector([0.0, 0.0, math.pi])
        else:
            sky_direction_scene = "UP"
            cam_to_rotated_q = quaternion.quaternion(1, 0, 0, 0)
    cam_to_rotated = np.eye(4)
    cam_to_rotated[:3, :3] = quaternion.as_rotation_matrix(cam_to_rotated_q)
    rotated_to_cam = np.linalg.inv(cam_to_rotated)
    return sky_direction_scene, rotated_to_cam


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.arkitscenes_dir, args.output_dir, args.num_workers)
