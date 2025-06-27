import os
from typing import Sequence
import copy
import numpy as np
import re
import datasets.mmact.constants as constants
from util.preprocessing.data_loader import SequenceStructure, NumpyLoader, RGBVideoLoader
from util.preprocessing.file_meta_data import FileMetaData


def get_file_metadata(root: str, rel_root: str, name: str) -> FileMetaData:
    possible_attributes = ("subject", "scene", "cam", "session")
    # possible_attributes = "subject"
    # 将路径拆开
    split_attributes = rel_root.split(os.path.sep)
    attributes = {}

    for s_a in split_attributes:
        for p_a in possible_attributes:
            if s_a.startswith(p_a):
                attributes[p_a] = int(s_a[len(p_a):]) - 1
                break

    assert "subject" in attributes

    action = constants.action_to_index_map[os.path.splitext(name)[0].lower()]
    fn = os.path.join(root, name)
    return FileMetaData(fn, action=action, **attributes)


# def get_file_metadata(root: str, rel_root: str, name: str) -> FileMetaData:
#     # 正则表达式匹配 'PDFE' 开头后跟数字和下划线和数字的格式
#     match = re.match(r'PDFE(\d+)_(\d+)', os.path.basename(rel_root))
#     if not match:
#         raise ValueError("路径格式不正确，应为 'PDFE' 开头后跟数字、下划线和数字")
#
#     # 从正则表达式匹配中提取 subject 和 session
#     subject = int(match.group(1))
#     session = int(match.group(2))
#
#     # 从文件名中提取动作名称，假设动作名称是文件名中的第一个部分，用 '_' 分隔
#     action_name = os.path.splitext(name)[0].split('_')[0]
#     if action_name not in ["0", "1"]:
#         raise ValueError(f"动作名称 '{action_name}' 不是有效的动作标识")
#     action_index = int(action_name)  # 将动作名称转换为整数索引
#
#     # 构建完整的文件路径
#     fn = os.path.join(root, name)
#
#     # 创建属性字典
#     attributes = {
#         'subject': subject,
#         'session': session
#     }
#
#     # 以指定的返回形式创建并返回 FileMetaData 实例
#     return FileMetaData(fn, action=action_index, **attributes)
def _is_valid_file(file: str) -> bool:
    ext = os.path.splitext(file)[1]
    return ext in (".csv", ".mp4", ".npy")


def get_files(data_path: str, repeat_view: int = 0) -> Sequence[FileMetaData]:
    out_files = []
    for root, _, files in os.walk(data_path, followlinks=True):
        rel_root = os.path.relpath(root, data_path)

        for name in files:
            if _is_valid_file(name):
                file = get_file_metadata(root, rel_root, name)
                out_files.append(file)
                if repeat_view > 1:
                    file.properties["cam"] = 0
                    setattr(file, "cam", 0)
                    for i in range(1, repeat_view):
                        file2 = copy.deepcopy(file)
                        file2.properties["cam"] = i
                        setattr(file2, "cam", i)
                        out_files.append(file2)

    return out_files
# def get_files(data_path: str, repeat_view: int = 0) -> Sequence[FileMetaData]:
#     out_files = []
#     for root, _, files in os.walk(data_path, followlinks=True):
#         rel_root = os.path.relpath(root, data_path)
#
#         for name in files:
#             if _is_valid_file(name):
#                 file_metadata = get_file_metadata(root, rel_root, name)
#                 out_files.append(file_metadata)
#                 if repeat_view > 1:
#                     # 创建重复视图的副本
#                     for i in range(1, repeat_view):
#                         file_copy = copy.deepcopy(file_metadata)
#                         # 假设 'cam' 是我们要重复的属性
#                         file_copy.properties["cam"] = i
#                         setattr(file_copy, "cam", i)  # 如果需要在实例中也设置属性
#                         out_files.append(file_copy)
#
#     return out_files

def get_classes(data_path: str) -> Sequence[str]:
    classes = set()
    for _, _, files in os.walk(data_path):
        for file in files:
            name = os.path.splitext(file)[0]
            classes.add(name.lower())
    classes = list(sorted(classes))
    return classes


skeleton_sequence_structure = SequenceStructure(constants.skeleton_rgb_max_sequence_length, constants.skeleton_shape,
                                                np.float32)
rgb_sequence_structure = SequenceStructure(constants.skeleton_rgb_max_sequence_length, constants.rgb_shape, np.uint8)

skeleton_loader = NumpyLoader("skeleton", skeleton_sequence_structure)
rgb_loader = RGBVideoLoader("rgb", rgb_sequence_structure)
