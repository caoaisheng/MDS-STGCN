import argparse
import os
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

import datasets.mmact.constants as constants
import datasets.mmact.lo as io
from datasets.mmact.config import get_preprocessing_setting
from util.dynamic_import import import_class
from util.merge import deep_merge_dictionary
from util.preprocessing.data_loader import SequenceStructure, NumpyLoader
from util.preprocessing.datagroup import DataGroup


def get_configuration() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="data data conversion.")
    parser.add_argument("-i", "--in_path", default="E:/Fogvideo/30-fold-unpre", type=str,
                        help="MMAct data parent directory")
    parser.add_argument("-o", "--out_path", default="E:/Fogvideo/preprocessed_data/30-fold/30", type=str,
                        help="Destination directory for processed data.")
    parser.add_argument("-m", "--modes", type=str, help="Modes (comma-separated) to decide how to process the dataset."
                                                        " See config.py for all modes.")
    #决定如何处理数据集的模式（用逗号分隔），具体模式可以参考config.py文件
    parser.add_argument("-t", "--target_modality", type=str,
                        help="Name of a modality. "
                             "All sequences are resampled to be of the "
                             "maximum sequence length of the specified modality.")
    parser.add_argument("--split", default="cross_subject", type=str, choices=("cross_subject", "cross_view"),
                        help="Which split to use for training and test data.")
    parser.add_argument("--shrink", default=1, type=int,
                        help="Shrink sequence length by this factor. "
                             "E.g. skeleton/rgb are captured with 30FPS -> Reduce to 10FPS")
    parser.add_argument("-w", "--wearable_sensors", nargs="+",
                        default=("gyro_clip",  "acc_watch_clip"),
                        help="Which wearable sensor modalities to use. "
                             "The order is important: Resample the length of all other sensor modalities "
                             "to that of the first element in this list.")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    return parser.parse_args()




def create_labels(out_path: str, data_group: DataGroup, splits: dict):
    """
    Create labels and write them to files

    :param out_path: path where label files will be stored
    :param data_group: data group to create labels from
    :param splits: dataset splits
    :param split_type: split type
    """
    label_splits = data_group.produce_labels(splits)
    for split_name, labels in label_splits.items():
        np.save(os.path.join(out_path, f"{split_name}_labels.npy"), labels)


def filter_file_list(modalities: list) -> list:
    if len(modalities) == 1:
        return modalities

    index_tuples = []
    for main_idx, main_file in enumerate(tqdm(modalities[0][1],
                                              desc="Multiple modalities. 过滤所有模态中没有样本的文件"
                                                   "Filter files without sample for all modalities")):
        a = [-1] * len(modalities) # -1表示没找到对应动作的文件
        a[0] = main_idx
        for i in range(1, len(modalities)):
            for idx, file in enumerate(modalities[i][1]):
                if main_file.is_same_action(file):
                    a[i] = idx
                    break
        if -1 not in a:
            index_tuples.append(a)

    # move all files to new lists of equal size
    new_files = []
    for _ in modalities:
        new_files.append([])
    for t in index_tuples:
        for i in range(len(t)):
            new_files[i].append(modalities[i][1][t[i]])

    for i in range(1, len(new_files)):
        assert len(new_files[i - 1]) == len(new_files[i])

    for i in range(len(modalities)):
        modalities[i] = (modalities[i][0], new_files[i])

    return modalities


def preprocess(cf: argparse.Namespace):
    # dataset splits
    if cf.split == "cross_subject":
        splits = {
            "train": constants.cross_subject_training,
            "val": constants.cross_subject_test
        }
        # split_type = "subject"
    elif cf.split == "cross_view":
        splits = {
            "train": constants.cross_view_training,
            "val": constants.cross_view_test
        }
        split_type = "cam"
    else:
        raise ValueError("Unsupported split")

    modes = ["skeleton_imu_enhanced"] if cf.modes is None else cf.modes.split(",")
    setting = deep_merge_dictionary((get_preprocessing_setting(mode) for mode in modes))

    if "kwargs" not in setting:
        setting["kwargs"] = {}
    setting["kwargs"]["num_bodies"] = 1
    if "imu_num_signals" in setting["kwargs"]:
        setting["kwargs"]["imu_num_signals"] = len(cf.wearable_sensors)

    if cf.debug:
        setting["kwargs"].update({
            "debug": True,
            # "skeleton_edges": skeleton_edges,
            "actions": constants.actions,
            # "skeleton_joint_labels": skeleton_joints
        })

    # which data processors to use (will be dynamically loaded from util.preprocessing.processor)
    processors = setting["processors"]
    processors = {k: import_class(f"util.preprocessing.processor.{v}") for k, v in processors.items()}

    # modes for each data processor
    processor_modes = setting.get("modes", None)

    subdir = "__".join(modes)
    out_path = os.path.join(cf.out_path, subdir, cf.split)
    os.makedirs(out_path, exist_ok=True)

    modalities = []

    if "skeleton" in setting["input"]:
        skeleton_data_files = io.get_files(os.path.join(cf.in_path, "pose/30"))
        modalities.append((io.skeleton_loader, skeleton_data_files))
    if "rgb" in setting["input"]:
        rgb_data_files = io.get_files(os.path.join(cf.in_path, "RGB"))
        modalities.append((io.rgb_loader, rgb_data_files))
    if "inertial" in setting["input"]:
        # repeat signal data for every camera if view-dependent modalities are also loaded
        repeat_view = constants.num_views if ("rgb" in setting["input"] or "skeleton" in setting["input"]) else 0
        inertial_data_files = io.get_files(os.path.join(cf.in_path, "imu/30"), repeat_view=repeat_view)
        inertial_structure = SequenceStructure(constants.inertial_max_sequence_length,
                                               (constants.inertial_max_sequence_length, 3 * 2),
                                               np.float32)
        inertial_loader = NumpyLoader("inertial", inertial_structure)
        modalities.append((inertial_loader, inertial_data_files))

    modalities = filter_file_list(modalities)

    multi_modal_data_group = DataGroup.create(modalities)
    # create_labels(out_path, multi_modal_data_group, splits, split_type)
    create_labels(out_path, multi_modal_data_group, splits)
    # Create features for each modality and write them to files
    # Mode keys are equivalent to processor keys defined above to set the mode for a specific processor
    # multi_modal_data_group.produce_features(splits, processors=processors, main_modality=cf.target_modality,
    #                                         modes=processor_modes, out_path=out_path, split_type=split_type,
    #                                         **setting["kwargs"])
    multi_modal_data_group.produce_features(splits, processors=processors, main_modality=cf.target_modality,
                                            modes=processor_modes, out_path=out_path, **setting["kwargs"])
    if cf.shrink > 1:
        print(f"Shrinking feature sequence length by factor {cf.shrink}")
        for file in os.scandir(out_path):
            if "feature" not in file.name:
                continue
            print(f"Shrinking '{file.path}'...")
            os.rename(file.path, file.path + ".old")
            arr = np.load(file.path + ".old", mmap_mode="r")
            np.save(file.path, arr[:, :, ::cf.shrink])
            del arr
            os.remove(file.path + ".old")


if __name__ == "__main__":
    conf = get_configuration()
    # merge_signal_data(conf.in_path, conf.wearable_sensors)
    os.makedirs(conf.out_path, exist_ok=True)
    preprocess(conf)
