import os
from vggt.data.datasets.scannet import ScanNetDataset
from vggt.data.datasets.scannetpp import ScanNetppDataset
from vggt.data.datasets.arkitscenes import ARKitScenesDataset
from vggt.data.datasets.co3d import Co3dDataset
from cambrianp.datasets.utils.transforms import SeqColorJitter, ImgNorm


def load_3r_dataset(data_args, scene_ids_file, dataset_name, sample_mode='unified',
                    input_use_augs=None, rec_use_augs=None):

    if not hasattr(data_args, 'patch_size'):
        data_args.patch_size = 14  # Standard patch size for ViT

    if data_args.use_augs or input_use_augs or rec_use_augs:
        data_args.aug_scales = [0.8, 1.2] # the random resize scale is set here. Can't use list in DataArgs
    else:
        data_args.aug_scales = [1.0, 1.0]

    if dataset_name == "scannet":
        dataset = ScanNetDataset(
            data_args=data_args,
            split="train",
            SCANNET_DIR=os.path.join(data_args.data_path, "processed_scannet_f"),
            scene_ids_file=scene_ids_file,
            sample_mode=sample_mode,
            input_use_augs=input_use_augs,
            rec_use_augs=rec_use_augs,
        )
        return dataset

    elif dataset_name == "scannetpp":
        dataset = ScanNetppDataset(
            data_args=data_args,
            split="train",
            SCANNETPP_DIR=os.path.join(data_args.data_path, "scannetpp"),
            scene_ids_file=scene_ids_file,
            sample_mode=sample_mode,
            input_use_augs=input_use_augs,
            rec_use_augs=rec_use_augs,
        )
        return dataset

    elif dataset_name == "arkitscenes":
        dataset = ARKitScenesDataset(
            data_args=data_args,
            split="train",
            ARKITSCENES_DIR=os.path.join(data_args.data_path, "arkitscenes"),
            scene_ids_file=scene_ids_file,
            sample_mode=sample_mode,
            input_use_augs=input_use_augs,
            rec_use_augs=rec_use_augs,
        )
        return dataset

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
