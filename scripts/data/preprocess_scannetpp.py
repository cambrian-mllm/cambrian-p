#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Script to pre-process the scannet++ dataset.
# Usage:
# python3 scripts/data/preprocess_scannetpp.py --scannetpp_dir /path/to/scannetpp --pyopengl-platform egl
# --------------------------------------------------------
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import argparse
import os.path as osp
import re
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation
import pyrender
import trimesh
import trimesh.exchange.ply
import numpy as np
import cv2
import PIL.Image as Image
import glob
from datasets_preprocess.utils.cropping import rescale_image_depthmap
import dust3r.utils.geometry as geometry

inv = np.linalg.inv
norm = np.linalg.norm
REGEXPR_DSLR = re.compile(r"^DSC(?P<frameid>\d+).JPG$")
REGEXPR_IPHONE =  re.compile(r"frame_(?P<frameid>\d+).jpg$")

DEBUG_VIZ = None  # 'iou'
if DEBUG_VIZ is not None:
    import matplotlib.pyplot as plt  # noqa


OPENGL_TO_OPENCV = np.float32(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannetpp_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--target_resolution", default=920, type=int, help="images resolution"
    )
    parser.add_argument(
        "--pyopengl-platform", type=str, default="", help="PyOpenGL env variable"
    )
    return parser


def pose_from_qwxyz_txyz(elems):
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(pose)  # returns cam2world


def get_frame_number(name, cam_type="dslr"):
    if cam_type == "dslr":
        regex_expr = REGEXPR_DSLR
    elif cam_type == "iphone":
        regex_expr = REGEXPR_IPHONE
    else:
        raise NotImplementedError(f"wrong {cam_type=} for get_frame_number")

    try:
        matches = re.match(regex_expr, name)
        if matches:
            frame_id = matches["frameid"]
        else:
            return int(frame_id)
            print(f"No regex match for {name} with cam_type={cam_type}")
            return 0
    except Exception as e:
        print(f"Error parsing frame number from {name}: {e}")
        return 0


def load_sfm(sfm_dir, cam_type="dslr"):
    try:
        # load cameras
        cameras_path = osp.join(sfm_dir, "cameras.txt")
        if not osp.exists(cameras_path):
            raise FileNotFoundError(f"cameras.txt not found in {sfm_dir}")

        with open(cameras_path, "r") as f:
            raw = f.read().splitlines()[3:]  # skip header

        intrinsics = {}
        for camera in tqdm(raw, position=1, leave=False, desc="Loading cameras"):
            camera = camera.split(" ")
            intrinsics[int(camera[0])] = [camera[1]] + [float(cam) for cam in camera[2:]]

        # load images
        images_path = osp.join(sfm_dir, "images.txt")
        if not osp.exists(images_path):
            raise FileNotFoundError(f"images.txt not found in {sfm_dir}")

        with open(images_path, "r") as f:
            raw = f.read().splitlines()
            raw = [line for line in raw if not line.startswith("#")]  # skip header

        img_idx = {}
        img_infos = {}
        for image, points in tqdm(
            zip(raw[0::2], raw[1::2]), total=len(raw) // 2, position=1, leave=False, desc="Loading images"
        ):
            try:
                image = image.split(" ")
                points = points.split(" ")

                idx = len(img_infos)

                img_name = image[-1]
                if img_name.startswith('iphone/'):
                    img_name = img_name.split('/', 1)[1]
                if img_name.startswith('video/'):
                    img_name = img_name.split('/', 1)[1]

                if img_name in img_idx:
                    print(f"Warning: duplicate db image: {img_name}")
                    continue

                img_idx[img_name] = idx  # register image name

                current_points2D = {}
                try:
                    for i, x, y in zip(points[2::3], points[0::3], points[1::3]):
                        if i != "-1":
                            current_points2D[int(i)] = (float(x), float(y))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing 2D points for {img_name}: {e}")
                    current_points2D = {}

                try:
                    cam_to_world = pose_from_qwxyz_txyz(image[1:-2])
                    frame_id = get_frame_number(img_name, cam_type)

                    img_infos[idx] = dict(
                        intrinsics=intrinsics[int(image[-2])],
                        path=img_name,
                        frame_id=frame_id,
                        cam_to_world=cam_to_world,
                        sparse_pts2d=current_points2D,
                    )
                except Exception as e:
                    print(f"Error processing image {img_name}: {e}")
                    continue

            except Exception as e:
                print(f"Error processing image line: {e}")
                continue

        # load 3D points
        points3d_path = osp.join(sfm_dir, "points3D.txt")
        if not osp.exists(points3d_path):
            print(f"Warning: points3D.txt not found in {sfm_dir}")
            return img_idx, img_infos, {}, {idx: [] for idx in img_infos.keys()}

        with open(points3d_path, "r") as f:
            raw = f.read().splitlines()
            raw = [line for line in raw if not line.startswith("#")]  # skip header

        points3D = {}
        observations = {idx: [] for idx in img_infos.keys()}
        for point in tqdm(raw, position=1, leave=False, desc="Loading 3D points"):
            try:
                point = point.split()
                point_3d_idx = int(point[0])
                points3D[point_3d_idx] = tuple(map(float, point[1:4]))
                if len(point) > 8:
                    for idx, point_2d_idx in zip(point[8::2], point[9::2]):
                        if idx in observations:
                            observations[idx].append((point_3d_idx, int(point_2d_idx)))
            except Exception as e:
                print(f"Warning: Error processing 3D point: {e}")
                continue

        return img_idx, img_infos, points3D, observations

    except Exception as e:
        print(f"Error in load_sfm for {sfm_dir}: {e}")
        raise


def sample_frames_uniformly(img_infos, num_frames):
    """Sample frames uniformly based on frame numbers (only frame_ images)"""
    # Filter and extract frame numbers from frame_ images
    frame_data = []
    for idx, val in img_infos.items():
        if val["path"].startswith("frame_"):
            try:
                # Extract frame number from "frame_XXXXXX.jpg" -> XXXXXX
                frame_num = int(val["path"].split(".")[0].split("_")[-1])
                frame_data.append((frame_num, idx, val))
            except (ValueError, IndexError):
                print(f"Could not parse frame number from {val['path']}")
                continue

    if len(frame_data) == 0:
        return {}

    # Sort by frame number
    frame_data.sort(key=lambda x: x[0])

    if num_frames is None or len(frame_data) <= num_frames:
        return {idx: val for _, idx, val in frame_data}

    # Sample uniformly in frame number space
    frame_numbers = [frame_num for frame_num, _, _ in frame_data]
    min_frame = min(frame_numbers)
    max_frame = max(frame_numbers)

    # Generate uniform frame numbers
    uniform_frame_numbers = np.linspace(min_frame, max_frame, num_frames).astype(int)

    # Find closest available frames for each uniform sample
    selected_data = []
    used_frame_numbers = set()

    for target_frame in uniform_frame_numbers:
        # Find closest available frame number that hasn't been used
        best_data = None
        best_distance = float('inf')

        for frame_num, idx, val in frame_data:
            if frame_num not in used_frame_numbers:
                distance = abs(frame_num - target_frame)
                if distance < best_distance:
                    best_distance = distance
                    best_data = (frame_num, idx, val)

        if best_data is not None:
            selected_data.append(best_data)
            used_frame_numbers.add(best_data[0])

    return {idx: val for _, idx, val in selected_data[:num_frames]}


def undistort_images(intrinsics, rgb, mask):
    camera_type = intrinsics[0]

    width = int(intrinsics[1])
    height = int(intrinsics[2])
    fx = intrinsics[3]
    fy = intrinsics[4]
    cx = intrinsics[5]
    cy = intrinsics[6]
    distortion = np.array(intrinsics[7:])

    K = np.zeros([3, 3])
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    K[2, 2] = 1

    K = geometry.colmap_to_opencv_intrinsics(K)
    if camera_type == "OPENCV_FISHEYE":
        assert len(distortion) == 4

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            distortion,
            (width, height),
            np.eye(3),
            balance=0.0,
        )
        # Make the cx and cy to be the center of the image
        new_K[0, 2] = width / 2.0
        new_K[1, 2] = height / 2.0

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1
        )
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K, distortion, (width, height), 1, (width, height), True
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1
        )

    undistorted_image = cv2.remap(
        rgb,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    undistorted_mask = cv2.remap(
        mask,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    K = geometry.opencv_to_colmap_intrinsics(K)
    return width, height, new_K, undistorted_image, undistorted_mask


def discover_scenes(root_dir):
    """Discover all available scenes in the ScanNet++ dataset"""
    data_dir = osp.join(root_dir, "data")
    if not osp.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")

    scenes = []
    for scene_name in os.listdir(data_dir):
        scene_path = osp.join(data_dir, scene_name)
        if not osp.isdir(scene_path):
            continue

        # Check if scene has required subdirectories (only iPhone needed now)
        iphone_dir = osp.join(scene_path, "iphone")
        scans_dir = osp.join(scene_path, "scans")

        if osp.exists(iphone_dir) and osp.exists(scans_dir):
            scenes.append(scene_name)

    return sorted(scenes)


def generate_pairs(num_frames):
    """Generate all possible pairs for the selected frames"""
    pairs = []
    for i in range(num_frames):
        for j in range(i + 1, num_frames):
            pairs.append([i, j])
    return np.array(pairs)


def process_scene_data(scene, selected_img_infos, mesh_scene, renderer, target_resolution,
                      rgb_dir_iphone, mask_dir_iphone, output_dir_scene_rgb, output_dir_scene_depth,
                      znear=0.05, zfar=20.0):
    """Process scene data and return metadata for given selection."""

    mesh = pyrender.Mesh.from_trimesh(mesh_scene, smooth=False)
    pyrender_scene = pyrender.Scene()
    pyrender_scene.add(mesh)

    # Process selected frames
    trajectories = []
    intrinsics = []
    images = []

    for combined_idx, img_info in tqdm(selected_img_infos.items(), position=1, leave=False):
        try:
            rgb = np.array(Image.open(os.path.join(rgb_dir_iphone, img_info["path"])))
            mask_path = os.path.join(mask_dir_iphone, img_info["path"][:-3] + "png")
            mask = np.array(Image.open(mask_path))

            _, _, K, rgb, mask = undistort_images(
                img_info["intrinsics"], rgb, mask
            )

            # rescale_image_depthmap assumes opencv intrinsics
            intrinsics_opencv = geometry.colmap_to_opencv_intrinsics(K)
            image, mask, intrinsics_opencv = rescale_image_depthmap(
                rgb,
                mask,
                intrinsics_opencv,
                (target_resolution, target_resolution * 3.0 / 4),
            )

            W, H = image.size
            intrinsics_colmap = geometry.opencv_to_colmap_intrinsics(intrinsics_opencv)

            # Save RGB image
            rgb_outpath = os.path.join(
                output_dir_scene_rgb, img_info["path"][:-3] + "jpg"
            )
            image.save(rgb_outpath)

            # Render and save depth
            depth_outpath = os.path.join(
                output_dir_scene_depth, img_info["path"][:-3] + "png"
            )

            renderer.viewport_width, renderer.viewport_height = W, H
            fx, fy, cx, cy = (
                intrinsics_colmap[0, 0],
                intrinsics_colmap[1, 1],
                intrinsics_colmap[0, 2],
                intrinsics_colmap[1, 2],
            )
            camera = pyrender.camera.IntrinsicsCamera(
                fx, fy, cx, cy, znear=znear, zfar=zfar
            )
            camera_node = pyrender_scene.add(
                camera, pose=img_info["cam_to_world"] @ OPENGL_TO_OPENCV
            )

            depth = renderer.render(
                pyrender_scene, flags=pyrender.RenderFlags.DEPTH_ONLY
            )
            pyrender_scene.remove_node(camera_node)

            depth = (depth * 1000).astype("uint16")
            depth_mask = mask < 255
            depth[depth_mask] = 0
            Image.fromarray(depth).save(depth_outpath)

            # Store metadata
            trajectories.append(img_info["cam_to_world"])
            intrinsics.append(intrinsics_colmap)
            images.append(img_info["path"][:-4])  # Remove extension

        except Exception as e:
            print(f"Error processing image {img_info['path']} in scene {scene}: {e}")
            continue

    del pyrender_scene

    if len(trajectories) == 0:
        return None

    # Generate pairs for all selected frames
    pairs = generate_pairs(len(trajectories))

    # Convert to numpy arrays
    trajectories = np.stack(trajectories, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)
    images = np.array(images)

    return {
        'trajectories': trajectories,
        'intrinsics': intrinsics,
        'images': images,
        'pairs': pairs
    }


def process_scenes(root, output_dir, target_resolution):
    os.makedirs(output_dir, exist_ok=True)

    # default values from
    # https://github.com/scannetpp/scannetpp/blob/main/common/configs/render.yml
    znear = 0.05
    zfar = 20.0

    # Discover all scenes
    scenes = discover_scenes(root)
    print(f"Found {len(scenes)} scenes to process")

    # for each of these, we will select some iphone images
    # we will undistort them and render their depth
    renderer = pyrender.OffscreenRenderer(0, 0)

    for scene in tqdm(scenes, position=0, leave=True):
        data_dir = os.path.join(root, "data", scene)
        dir_iphone = os.path.join(data_dir, "iphone")
        dir_scans = os.path.join(data_dir, "scans")

        output_dir_scene = os.path.join(output_dir, scene)

        # Check if all sampling strategies are already processed
        sampling_configs = [
            ('uni32', 32),
            ('uni128', 128),
            ('all', None)
        ]

        all_processed = True
        for suffix, num_frames in sampling_configs:
            scene_metadata_path = osp.join(output_dir_scene, f"scene_metadata_{suffix}.npz")
            if not osp.isfile(scene_metadata_path):
                all_processed = False
                break

        if all_processed:
            print(f"Scene {scene} already processed for all sampling strategies, skipping...")
            continue

        # set up the output paths
        output_dir_scene_rgb = os.path.join(output_dir_scene, "images")
        output_dir_scene_depth = os.path.join(output_dir_scene, "depth")
        os.makedirs(output_dir_scene_rgb, exist_ok=True)
        os.makedirs(output_dir_scene_depth, exist_ok=True)

        ply_path = os.path.join(dir_scans, "mesh_aligned_0.05.ply")
        if not osp.exists(ply_path):
            print(f"Mesh file not found for scene {scene}, skipping...")
            continue

        sfm_dir_iphone = os.path.join(dir_iphone, "colmap")
        rgb_dir_iphone = os.path.join(dir_iphone, "rgb")
        mask_dir_iphone = os.path.join(dir_iphone, "rgb_masks")

        # Check if required directories exist (only iPhone needed now)
        if not all(osp.exists(d) for d in [sfm_dir_iphone, rgb_dir_iphone, mask_dir_iphone]):
            print(f"Missing iPhone directories for scene {scene}, skipping...")
            continue

        # load the mesh
        try:
            with open(ply_path, "rb") as f:
                mesh_kwargs = trimesh.exchange.ply.load_ply(f)
            mesh_scene = trimesh.Trimesh(**mesh_kwargs)
        except Exception as e:
            print(f"Error loading mesh for scene {scene}: {e}")
            continue

        # read colmap reconstruction - only iPhone data
        try:
            img_idx_iphone, img_infos_iphone, points3D_iphone, observations_iphone = (
                load_sfm(sfm_dir_iphone, cam_type="iphone")
            )
        except Exception as e:
            print(f"Error loading SfM data for scene {scene}: {e}")
            continue

        # Filter to only include frame_ images (iPhone images)
        frame_img_infos = {}
        for idx, info in img_infos_iphone.items():
            if info["path"].startswith("frame_"):
                frame_img_infos[f"iphone_{idx}"] = {**info, "camera_type": "iphone"}

        if len(frame_img_infos) == 0:
            print(f"No frame_ images found for scene {scene}, skipping...")
            continue

        scene_processed = False

        # Process different sampling strategies
        for suffix, num_frames in sampling_configs:
            scene_metadata_path = osp.join(output_dir_scene, f"scene_metadata_{suffix}.npz")

            if osp.isfile(scene_metadata_path):
                scene_processed = True
                continue

            # Sample frames based on strategy
            if num_frames is None:
                # Use all available frames
                selected_img_infos = sample_frames_uniformly(frame_img_infos, None)
                print(f"parsing {scene} - ALL {len(selected_img_infos)} frames from {len(frame_img_infos)} available")
            else:
                # Uniform sampling
                selected_img_infos = sample_frames_uniformly(frame_img_infos, num_frames)
                print(f"parsing {scene} - {len(selected_img_infos)} frames selected from {len(frame_img_infos)} available (uni{num_frames})")

            if len(selected_img_infos) == 0:
                print(f"No valid frames found for scene {scene} with {suffix}, skipping...")
                continue

            # Process scene data
            scene_data = process_scene_data(
                scene, selected_img_infos, mesh_scene, renderer, target_resolution,
                rgb_dir_iphone, mask_dir_iphone, output_dir_scene_rgb, output_dir_scene_depth,
                znear, zfar
            )

            if scene_data is None:
                print(f"No images successfully processed for scene {scene} with {suffix}")
                continue

            # Save scene metadata
            np.savez(
                scene_metadata_path,
                trajectories=scene_data['trajectories'],
                intrinsics=scene_data['intrinsics'],
                images=scene_data['images'],
                pairs=scene_data['pairs'],
            )

            scene_processed = True

        if scene_processed:
            print(f"Processed scene {scene}")

    # Create combined metadata for each sampling strategy
    for suffix, num_frames in sampling_configs:
        # Concatenate all scene metadata into a single file
        scene_data = {}
        processed_scenes = []

        for scene in scenes:
            scene_metadata_path = osp.join(output_dir, scene, f"scene_metadata_{suffix}.npz")
            if osp.exists(scene_metadata_path):
                try:
                    with np.load(scene_metadata_path) as data:
                        scene_data[scene] = {
                            "trajectories": data["trajectories"],
                            "intrinsics": data["intrinsics"],
                            "images": data["images"],
                            "pairs": data["pairs"],
                        }
                    processed_scenes.append(scene)
                except Exception as e:
                    print(f"Error loading metadata for scene {scene}: {e}")

        if not processed_scenes:
            print(f"No scenes were successfully processed for {suffix}")
            continue

        # Create global metadata
        offset = 0
        counts = []
        scenes_list = []
        sceneids = []
        images_all = []
        intrinsics_all = []
        trajectories_all = []
        pairs_all = []

        for scene_idx, scene in enumerate(processed_scenes):
            data = scene_data[scene]
            num_imgs = len(data["images"])

            scenes_list.append(scene)
            sceneids.extend([scene_idx] * num_imgs)

            images_all.append(data["images"])
            intrinsics_all.append(data["intrinsics"])
            trajectories_all.append(data["trajectories"])

            # Offset pairs to global indices
            scene_pairs = data["pairs"].copy()
            scene_pairs[:, 0:2] += offset
            pairs_all.append(scene_pairs)

            counts.append(offset)
            offset += num_imgs

        # Concatenate all data
        images_all = np.concatenate(images_all, axis=0)
        intrinsics_all = np.concatenate(intrinsics_all, axis=0)
        trajectories_all = np.concatenate(trajectories_all, axis=0)
        pairs_all = np.concatenate(pairs_all, axis=0)

        # Save global metadata
        np.savez(
            osp.join(output_dir, f"all_metadata_{suffix}.npz"),
            counts=counts,
            scenes=scenes_list,
            sceneids=sceneids,
            images=images_all,
            intrinsics=intrinsics_all,
            trajectories=trajectories_all,
            pairs=pairs_all,
        )

        print(f"Processing complete for {suffix}! Processed {len(processed_scenes)} scenes with {len(images_all)} total images")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.pyopengl_platform.strip():
        os.environ["PYOPENGL_PLATFORM"] = args.pyopengl_platform
    process_scenes(
        args.scannetpp_dir,
        args.output_dir,
        args.target_resolution,
    )
