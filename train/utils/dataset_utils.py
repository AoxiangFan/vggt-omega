import json
import os
from pathlib import Path


def _import_dataset(name):
    if name == "scannet":
        from datasets.scannet_dataset import ScannetDataset
        return ScannetDataset
    if name == "vdr":
        from datasets.vdr_dataset import VDRDataset
        return VDRDataset
    if name == "colmap":
        from datasets.colmap_dataset import ColmapDataset
        return ColmapDataset
    if name == "hypersim":
        from datasets.hypersim import HypersimDataset
        return HypersimDataset
    if name == "tartanair":
        from datasets.tartanair import TartanAirDataset
        return TartanAirDataset
    if name == "blendedmvg":
        from datasets.blendedmvg import BlendedMVGDataset
        return BlendedMVGDataset
    if name == "dynamic_replica":
        from datasets.dynamic_replica import DynamicReplicaDataset
        return DynamicReplicaDataset
    if name == "matrix_city":
        from datasets.matrix_city import MatrixCityDataset
        return MatrixCityDataset
    if name == "vkitti":
        from datasets.vkitti import VirtualKITTIDataset
        return VirtualKITTIDataset
    if name == "sailvos3d":
        from datasets.sailvos3d import SAILVOS3DDataset
        return SAILVOS3DDataset
    if name == "mvssynth":
        from datasets.mvssynth import MVSSynthDataset
        return MVSSynthDataset
    if name == "nerf":
        from datasets.nerf_dataset import NeRFDataset
        return NeRFDataset
    if name == "nerfstudio":
        from datasets.nerfstudio_dataset import NerfStudioDataset
        return NerfStudioDataset
    raise ValueError(f"Not a recognized dataset: {name}")


def get_dataset(dataset_name, split_filepath, single_debug_scan_id=None, verbose=True):
    """Return (dataset_class, scans) for the given dataset name.

    split_filepath: path to a text/JSON file listing scan IDs, resolved
        relative to $PWD.
    single_debug_scan_id: if set, overrides the split file with a single scan.
    """
    dataset_class = _import_dataset(dataset_name)

    split_filepath = Path(os.environ["PWD"]) / split_filepath

    # Datasets that use a JSON split file
    if dataset_name in ("hypersim", "matrix_city"):
        with open(split_filepath, "r") as f:
            data = json.load(f)
        scans = list(data.keys())
    else:
        with open(split_filepath) as f:
            scans = [line.strip() for line in f.readlines()]

    if single_debug_scan_id is not None:
        scans = [single_debug_scan_id]

    if verbose:
        print(f" {dataset_name} — {len(scans)} scan(s) ".center(80, "#"))

    return dataset_class, scans
