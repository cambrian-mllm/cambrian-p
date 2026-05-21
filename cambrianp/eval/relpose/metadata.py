import os

# Keep Minimal data directory.
# Dataset roots can be overridden via env vars:
#   CAMBRIANP_SCANNET_PATH, CAMBRIANP_TUM_PATH,
#   CAMBRIANP_SINTEL_PATH, CAMBRIANP_SINTEL_ANNO_PATH,
#   CAMBRIANP_VSI_CANONICAL_PATH, CAMBRIANP_VSI_VIDEOS_PATH,
#   CAMBRIANP_SCANNETPP_PATH
dataset_metadata = {
    "scannet": {
        "img_path": os.environ.get("CAMBRIANP_SCANNET_PATH", "/path/to/scannet"),
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_all"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_all.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
    },
    "tum": {
        "img_path": os.environ.get("CAMBRIANP_TUM_PATH", "/path/to/tum"),
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "rgb_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "groundtruth_90.txt"
        ),
        "traj_format": "tum",
        "seq_list": None,
        "full_seq": True,
    },
    "sintel": {
        "img_path": os.environ.get("CAMBRIANP_SINTEL_PATH", "/path/to/sintel/training/final"),
        "anno_path": os.environ.get("CAMBRIANP_SINTEL_ANNO_PATH", "/path/to/sintel/training/camdata_left"),
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(anno_path, seq),
        "traj_format": None,
        "seq_list": [
            "alley_2", "ambush_4", "ambush_5", "ambush_6",
            "cave_2", "cave_4",
            "market_2", "market_5", "market_6",
            "shaman_3", "sleeping_1", "sleeping_2",
            "temple_2", "temple_3",
        ],
        "full_seq": False,
    },
    # 90-frame canonical prep for *every* VSI bundle (ScanNet + ARKits + SPP),
    # so the MonST3R protocol kicks in: uniform 90 frames from each bundle's
    # video, paired with the corresponding gt.poses subset. Each bundle dir
    # contains {color_90/, pose_90.txt}.
    "vsi_canonical_90": {
        "img_path": os.environ.get("CAMBRIANP_VSI_CANONICAL_PATH",
                                   "/path/to/vsi_canonical_eval"),
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(img_path, seq, "pose_90.txt"),
        "traj_format": "replica",
        "seq_list": None,   # auto-discover whatever's prepped
        "full_seq": True,
    },
    # VSI bundles (ScanNet + ARKitScenes) prepped from each bundle's video.mp4
    # + gt.poses in trajectory.json. Names are the bundle ids (with task
    # suffix), so the merge can write directly into scenes_vsi/<bundle>/.
    "vsi_videos": {
        "img_path": os.environ.get("CAMBRIANP_VSI_VIDEOS_PATH", "/path/to/vsi_pose_eval"),
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_all"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(img_path, seq, "pose_all.txt"),
        "traj_format": "replica",
        "seq_list": [
            # ScanNet (13 bundles)
            "scene0149_00_route", "scene0203_01_counting", "scene0277_02_absdist",
            "scene0304_00_reldir", "scene0307_02_reldir",
            "scene0316_00_route", "scene0316_00_size",
            "scene0353_00_appearance", "scene0378_01",
            "scene0435_02_counting", "scene0488_01_route",
            "scene0565_00_counting",
            "scene0645_00_counting", "scene0645_00_counting_table",
            "scene0663_00_counting",
            # ARKitScenes (11 bundles)
            "arkits_42446103_absdist", "arkits_42446167_size",
            "arkits_42897647_size", "arkits_42897688_reldir",
            "arkits_45261121_absdist", "arkits_47204578_reldir",
            "arkits_47331262_route", "arkits_47333932_counting",
            "arkits_47429904_counting", "arkits_47429977_counting",
            "arkits_47430038_absdist",
        ],
        "full_seq": True,
    },
    # ScanNet++: 39 VSI-Bench all_wins scenes prepped from iPhone COLMAP poses.
    # color_all/ has symlinks to the COLMAP-tracked iPhone jpgs (a ~1/10 subset
    # of the native 30fps capture); pose_all.txt has the matching 4x4 c2w
    # (row-major, 16 floats per line), already temporally sorted.
    "scannetpp": {
        "img_path": os.environ.get("CAMBRIANP_SCANNETPP_PATH", "/path/to/scannetpp_pose_eval"),
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_all"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(img_path, seq, "pose_all.txt"),
        "traj_format": "replica",
        "seq_list": [
            "09c1414f1b", "0d2ee665be", "1ada7a0617", "21d970d8de", "25f3b7a318",
            "27dd4da69e", "286b55a2bf", "31a2c91c43", "3864514494", "38d58a7a31",
            "3db0a1c8f3", "3e8bba0176", "3f15a9266d", "45b0dac5e3", "5748ce6f01",
            "578511c8a9", "5942004064", "5eb31827b7", "5ee7c22ba0", "5f99900f09",
            "6115eddb86", "7b6477cb95", "825d228aec", "9071e139d9", "a24f64f7fb",
            "a8bf42d646", "ac48a9b736", "acd95847c5", "bcd2436daf", "bde1e479ad",
            "c49a8c6cff", "c4c04e6d6c", "c50d2d1d42", "cc5237fd77", "d755b3d9d8",
            "e398684d27", "f2dc06b1d2", "f3d64c30f8", "f9f95681fd",
        ],
        "full_seq": True,
    },
}
