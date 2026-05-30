# encoding: utf-8
"""KITchen dataset reference information."""

import os.path as osp

import mmcv
import numpy as np


cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
data_root = osp.join(root_dir, "datasets")
dataset_root = osp.join(data_root, "KITchen")
model_dir = osp.join(dataset_root, "models", "models")
objects_file_candidates = [
    osp.join(dataset_root, "object_names.txt"),
    osp.join(dataset_root, "objects.txt"),
    osp.join(dataset_root, "object_names.json"),
]


def _load_objects():
    for path in objects_file_candidates:
        if not osp.exists(path):
            continue
        if path.endswith(".json"):
            data = mmcv.load(path)
            if isinstance(data, list):
                return [str(item) for item in data]
            if isinstance(data, dict) and "objects" in data:
                return [str(item) for item in data["objects"]]
            continue
        with open(path, "r", encoding="utf-8") as handle:
            items = [line.strip() for line in handle.readlines()]
        return [item for item in items if item]
    return []


objects = _load_objects()
id2obj = {idx + 1: obj_name for idx, obj_name in enumerate(objects)}
obj2id = {obj_name: obj_id for obj_id, obj_name in id2obj.items()}
obj_num = len(objects)
vertex_scale = 0.001
width = 640
height = 480
zNear = 0.25
zFar = 6.0
camera_matrix = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def get_models_info():
    models_info_path = osp.join(model_dir, "models_info.json")
    if not osp.exists(models_info_path):
        return {}
    return mmcv.load(models_info_path)


def get_fps_points():
    fps_points_path = osp.join(model_dir, "fps_points.pkl")
    if not osp.exists(fps_points_path):
        return {}
    return mmcv.load(fps_points_path)


def get_keypoints_3d():
    keypoints_3d_path = osp.join(model_dir, "keypoints_3d.pkl")
    if not osp.exists(keypoints_3d_path):
        return {}
    return mmcv.load(keypoints_3d_path)
