import argparse
import dataclasses
import os
from dataclasses import dataclass
from typing import List, Optional
import yaml


@dataclass
class DataOptions:
    # which dataset to use
    dataset: str = "scannet"

    # base dataset path
    dataset_path: str = "/datasets/scannetv2"

    # where to look for a tuple file
    tuple_info_file_location: str = "data_splits/ScanNetv2/standard_split/"

    # suffix of a tuple filename
    mv_tuple_file_suffix: str = "_eight_view_deepvmvs.txt"

    # file listing scans to use
    dataset_scan_split_file: str = "data_splits/ScanNetv2/standard_split/scannetv2_train.txt"

    split: str = "train"


@dataclass
class Options:
    """Dataclass for housing experiment flags."""

    debug: bool = False
    random_seed: int = 0

    ################################### logs ###################################
    name: str = "debug"
    log_dir: str = os.path.join(os.path.expanduser("~"), "tmp/tensorboard")
    notes: str = ""
    log_interval: int = 100
    val_interval: int = 1000
    val_batches: int = 100

    ################################### data ###################################
    datasets: List[DataOptions] = None
    val_datasets: List[DataOptions] = None
    num_workers: int = 12

    # number of views per tuple
    num_images_in_tuple: int = 8

    image_width: int = 518
    image_height: int = 518
    val_image_width: int = 518
    val_image_height: int = 518

    single_output_format: bool = True
    high_res_validation: bool = False
    shuffle_tuple: bool = False

    # only run on a single scan (for debugging)
    single_debug_scan_id: str = None

    ############################# hyperparameters ##############################
    lr: float = 1e-4
    wd: float = 1e-4
    batch_size: int = 4
    val_batch_size: int = 4
    max_epochs: int = 16
    grad_clip_norm: float = 1.0
    num_sanity_val_steps: int = 0

    # number of GPUs
    gpus: int = 2

    # lightning precision string, e.g. "bf16-mixed", "16-mixed", 32
    precision: str = "bf16-mixed"

    # stepped LR schedule — LR drops by 10x at each step (in training steps)
    lr_steps: list = dataclasses.field(default_factory=lambda: [70000, 80000])

    ################################## model ###################################
    patch_size: int = 16
    embed_dim: int = 1024
    enable_camera: bool = True
    enable_depth: bool = True
    enable_alignment: bool = False
    use_sparse_index: bool = True
    use_triton_sparse: bool = False

    # lower LR multiplier for the pretrained patch_embed backbone
    backbone_lr_scale: float = 0.1

    ################################ checkpoints ###############################
    # resumes full Lightning training state (optimizer, scheduler, epoch, etc.)
    resume: str = None

    # loads full Lightning checkpoint (weights + training state)
    load_weights_from_checkpoint: str = None

    # lazy-loads only matching weights (strict=False, new params stay random)
    lazy_load_weights_from_checkpoint: str = None


class OptionsHandler:
    """Handles options files and optional CLI arguments."""

    def __init__(self, required_flags=[]):
        if required_flags is None:
            required_flags = []

        self.options = Options()
        self.required_flags = required_flags

        self.parser = argparse.ArgumentParser(description="VGGT-Omega Training Options")
        self.parser.add_argument("--config_file", type=str, default=None)
        self.parser.add_argument("--data_config_file", type=str, default=None)
        self.parser.add_argument("--val_data_config_file", type=str, default=None)

        self._populate_argparse()

    def parse_and_merge_options(self, config_filepaths=None, ignore_cl_args=False):
        if not ignore_cl_args:
            cl_args = self.parser.parse_args()

        if config_filepaths is not None:
            if isinstance(config_filepaths, list):
                for fp in config_filepaths:
                    self._merge_config_options(OptionsHandler.load_options_from_yaml(fp))
            else:
                self._merge_config_options(OptionsHandler.load_options_from_yaml(config_filepaths))
            self.config_filepaths = config_filepaths

        elif not ignore_cl_args and (
            cl_args.config_file is not None or cl_args.data_config_file is not None
        ):
            self.config_filepaths = []

            if cl_args.config_file is not None:
                self._merge_config_options(OptionsHandler.load_options_from_yaml(cl_args.config_file))
                self.config_filepaths.append(cl_args.config_file)

            if cl_args.data_config_file is not None:
                self.options.datasets = [
                    OptionsHandler.load_options_from_yaml(p)
                    for p in cl_args.data_config_file.split(":")
                ]
                self.config_filepaths.append(cl_args.data_config_file)

            if cl_args.val_data_config_file is not None:
                self.options.val_datasets = [
                    OptionsHandler.load_options_from_yaml(p)
                    for p in cl_args.val_data_config_file.split(":")
                ]
                self.config_filepaths.append(cl_args.val_data_config_file)
        else:
            print("Not reading from a config_file.")
            self.config_filepaths = None

        if not ignore_cl_args:
            self._merge_cl_args(cl_args)

        self._check_required_items()

    def _populate_argparse(self):
        for field_name, field_info in self.options.__dataclass_fields__.items():
            if field_info.type == bool:
                self.parser.add_argument(f"--{field_name}", action="store_true")
            else:
                self.parser.add_argument(f"--{field_name}", type=field_info.type, default=None)

    def _check_required_items(self):
        for flag in self.required_flags:
            if self.options.__getattribute__(flag) is None:
                raise Exception(f"Missing required config argument '{flag}'")

    def _merge_config_options(self, config_options):
        for field_name in config_options.__dict__.keys():
            self.options.__setattr__(field_name, config_options.__getattribute__(field_name))

    def _merge_cl_args(self, cl_args):
        for arg_pair in cl_args._get_kwargs():
            if arg_pair[0] in ("config_file", "data_config_file", "val_data_config_file"):
                continue
            if arg_pair[1] is not None:
                if isinstance(arg_pair[1], bool) and not arg_pair[1]:
                    continue
                self.options.__setattr__(arg_pair[0], arg_pair[1])

    def pretty_print_options(self):
        print("########################### Options ###########################")
        for field_name in self.options.__dataclass_fields__.keys():
            print("    ", field_name + ":", self.options.__getattribute__(field_name))
        print("###############################################################")

    @staticmethod
    def load_options_from_yaml(config_filepath):
        with open(config_filepath, "r") as stream:
            return yaml.load(stream, Loader=yaml.Loader)

    @staticmethod
    def save_options_as_yaml(config_filepath, options):
        with open(config_filepath, "w") as outfile:
            yaml.dump(options, outfile, default_flow_style=False)
