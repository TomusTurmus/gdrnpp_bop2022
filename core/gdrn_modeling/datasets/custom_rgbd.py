import glob
import hashlib
import logging
import os.path as osp
import sys
import time
from collections import OrderedDict

import mmcv
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from transforms3d.quaternions import mat2quat

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)

import ref
from lib.pysixd import inout, misc
from lib.utils.utils import lazy_property


logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))
DEFAULT_DATASET_ROOT = osp.join(DATASETS_ROOT, "custom_data_1")


def _expand(path):
    return osp.expanduser(path) if isinstance(path, str) else path


def _as_numpy_cam(cam):
    if cam is None:
        return None
    return np.array(cam, dtype=np.float32).reshape(3, 3)


class CUSTOM_RGBD_Dataset:
    """Dataset adapter for a folder of RGBD images.

    The loader accepts either a manifest file or a plain folder layout. When no
    annotations are present, it still returns image records so the detector can
    run and the pose stage can safely emit empty outputs.
    """

    def __init__(self, data_cfg):
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.dataset_root = _expand(data_cfg.get("dataset_root", DEFAULT_DATASET_ROOT))
        self.image_dir = _expand(data_cfg.get("image_dir", osp.join(self.dataset_root, "rgb")))
        self.depth_dir = _expand(data_cfg.get("depth_dir", osp.join(self.dataset_root, "depth")))
        self.camera_file = _expand(data_cfg.get("camera_file", osp.join(self.dataset_root, "camera.json")))
        self.ann_file = _expand(data_cfg.get("ann_file", ""))

        self.objs = data_cfg.get("objs", [])
        self.select_objs = data_cfg.get("select_objs", self.objs)
        self.ref_key = data_cfg.get("ref_key", "custom_data_1")

        if not self.objs and hasattr(ref, self.ref_key):
            self.objs = getattr(ref, self.ref_key).objects
        if not self.select_objs:
            self.select_objs = self.objs

        self.scale_to_meter = data_cfg.get("scale_to_meter", 0.001)
        self.with_masks = data_cfg.get("with_masks", False)
        self.with_depth = data_cfg.get("with_depth", True)
        self.height = data_cfg.get("height", 480)
        self.width = data_cfg.get("width", 640)
        self.cache_dir = _expand(data_cfg.get("cache_dir", osp.join(PROJ_ROOT, ".cache")))
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg.get("num_to_load", -1)
        self.filter_invalid = data_cfg.get("filter_invalid", True)
        self.scene_id = int(data_cfg.get("scene_id", 0))
        self.depth_scale = float(data_cfg.get("depth_scale", 1000.0))

        self.cat_ids = [cat_id for cat_id, obj_name in getattr(ref, self.ref_key).id2obj.items() if obj_name in self.objs]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs))

    def _load_camera_info(self):
        if self.camera_file and osp.exists(self.camera_file):
            return mmcv.load(self.camera_file)
        return None

    def _camera_for_image(self, camera_info, image_stem, image_index):
        if camera_info is None:
            return None, self.depth_scale
        if isinstance(camera_info, dict):
            if "scene_camera" in camera_info and isinstance(camera_info["scene_camera"], dict):
                scene_camera = camera_info["scene_camera"]
                info = scene_camera.get(str(image_index)) or scene_camera.get(f"{image_index:06d}") or camera_info
            else:
                info = camera_info.get(str(image_index)) if str(image_index) in camera_info else camera_info
        else:
            info = camera_info

        if isinstance(info, dict):
            cam = info.get("cam_K")
            if cam is None:
                cam = info.get("K")
            if cam is None:
                cam = info.get("camera_matrix")
            depth_scale = float(info.get("depth_scale", self.depth_scale))
        else:
            cam = info
            depth_scale = self.depth_scale
        return _as_numpy_cam(cam), depth_scale

    def _build_from_manifest(self, manifest):
        records = []
        if isinstance(manifest, dict):
            if "dataset_dicts" in manifest:
                manifest = manifest["dataset_dicts"]
            elif "samples" in manifest:
                manifest = manifest["samples"]
        if not isinstance(manifest, list):
            raise ValueError(f"Unsupported custom RGBD manifest type: {type(manifest)}")

        for image_index, record in enumerate(manifest):
            normalized = dict(record)
            file_name = normalized.get("file_name") or normalized.get("rgb_file") or normalized.get("image_file")
            if file_name is None:
                continue
            if not osp.isabs(file_name):
                file_name = osp.join(self.dataset_root, file_name)
            normalized["file_name"] = osp.relpath(file_name, PROJ_ROOT)

            depth_file = normalized.get("depth_file") or normalized.get("depth_path")
            if depth_file is not None and not osp.isabs(depth_file):
                depth_file = osp.join(self.dataset_root, depth_file)
            if depth_file is not None:
                normalized["depth_file"] = osp.relpath(depth_file, PROJ_ROOT)

            normalized.setdefault("height", self.height)
            normalized.setdefault("width", self.width)
            normalized.setdefault("scene_id", self.scene_id)
            normalized.setdefault("image_id", image_index)
            normalized.setdefault("scene_im_id", f"{normalized['scene_id']}/{normalized['image_id']}")
            normalized.setdefault("img_type", "real")

            cam = normalized.get("cam")
            if cam is None:
                cam = normalized.get("K")
            if cam is None:
                cam = normalized.get("camera_matrix")
            if cam is None:
                cam, depth_scale = self._camera_for_image(self._load_camera_info(), osp.splitext(osp.basename(file_name))[0], image_index)
            else:
                cam = _as_numpy_cam(cam)
                depth_scale = float(normalized.get("depth_scale", self.depth_scale))
            if cam is not None:
                normalized["cam"] = cam
            normalized["depth_factor"] = 1000.0 / depth_scale

            annos = normalized.get("annotations") or normalized.get("annos") or []
            normalized_annos = []
            for anno in annos:
                inst = dict(anno)
                if "bbox_mode" not in inst:
                    inst["bbox_mode"] = BoxMode.XYWH_ABS
                if "category_id" not in inst and self.objs:
                    obj_name = inst.get("obj_name") or inst.get("category_name")
                    if obj_name in self.obj2label:
                        inst["category_id"] = self.obj2label[obj_name]
                if self.filter_invalid and "bbox" in inst and "bbox_mode" in inst:
                    bbox = BoxMode.convert(inst["bbox"], inst["bbox_mode"], BoxMode.XYXY_ABS)
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                normalized_annos.append(inst)
            normalized["annotations"] = normalized_annos
            records.append(normalized)
        return records

    def _scan_image_folder(self):
        camera_info = self._load_camera_info()
        image_paths = []
        for ext in ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]:
            image_paths.extend(glob.glob(osp.join(self.image_dir, f"**/*.{ext}"), recursive=True))
        image_paths = sorted(set(image_paths))

        records = []
        for image_index, file_path in enumerate(image_paths):
            stem = osp.splitext(osp.basename(file_path))[0]
            depth_candidates = []
            for ext in ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]:
                depth_candidates.append(osp.join(self.depth_dir, f"{stem}.{ext}"))
                depth_candidates.append(osp.join(self.depth_dir, f"{stem}_depth.{ext}"))
            depth_file = next((path for path in depth_candidates if osp.exists(path)), None)

            cam, depth_scale = self._camera_for_image(camera_info, stem, image_index)
            if cam is None:
                cam = getattr(ref, self.ref_key).camera_matrix if hasattr(ref, self.ref_key) else np.eye(3, dtype=np.float32)

            record = {
                "dataset_name": self.name,
                "file_name": osp.relpath(file_path, PROJ_ROOT),
                "height": self.height,
                "width": self.width,
                "image_id": image_index,
                "scene_id": self.scene_id,
                "scene_im_id": f"{self.scene_id}/{image_index}",
                "cam": cam,
                "depth_factor": 1000.0 / depth_scale,
                "img_type": "real",
                "annotations": [],
            }
            if depth_file is not None:
                record["depth_file"] = osp.relpath(depth_file, PROJ_ROOT)
            if self.with_depth and depth_file is None:
                logger.warning("No depth image found for %s", file_path)
            records.append(record)
        return records

    def __call__(self):
        hashed_file_name = hashlib.md5(
            (
                "".join([str(fn) for fn in self.objs])
                + "dataset_dicts_{}_{}_{}_{}_{}".format(
                    self.name,
                    self.dataset_root,
                    self.with_masks,
                    self.with_depth,
                    __name__,
                )
            ).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(self.cache_dir, f"dataset_dicts_{self.name}_{hashed_file_name}.pkl")

        if osp.exists(cache_path) and self.use_cache:
            logger.info("load cached dataset dicts from %s", cache_path)
            return mmcv.load(cache_path)

        t_start = time.perf_counter()
        logger.info("loading dataset dicts: %s", self.name)

        if self.ann_file and osp.exists(self.ann_file):
            manifest = mmcv.load(self.ann_file)
            dataset_dicts = self._build_from_manifest(manifest)
        else:
            dataset_dicts = self._scan_image_folder()

        if self.num_to_load > 0:
            self.num_to_load = min(int(self.num_to_load), len(dataset_dicts))
            dataset_dicts = dataset_dicts[: self.num_to_load]

        logger.info("loaded %d dataset dicts, using %.2fs", len(dataset_dicts), time.perf_counter() - t_start)
        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(dataset_dicts, cache_path, protocol=4)
        logger.info("Dumped dataset_dicts to %s", cache_path)
        return dataset_dicts

    @lazy_property
    def models_info(self):
        models_info_path = osp.join(self.dataset_root, "models", "models_info.json")
        if not osp.exists(models_info_path):
            return {}
        return mmcv.load(models_info_path)

    @lazy_property
    def models(self):
        models_root = osp.join(self.dataset_root, "models")
        cache_path = osp.join(self.cache_dir, f"models_{self.name}.pkl")
        if osp.exists(cache_path) and self.use_cache:
            return mmcv.load(cache_path)

        models = []
        for obj_name in self.objs:
            obj_id = getattr(ref, self.ref_key).obj2id.get(obj_name, None) if hasattr(ref, self.ref_key) else None
            if obj_id is None:
                continue
            model_candidates = [
                osp.join(models_root, f"obj_{obj_id:06d}.ply"),
                osp.join(models_root, f"{obj_name}.ply"),
            ]
            model_path = next((path for path in model_candidates if osp.exists(path)), None)
            if model_path is None:
                continue
            model = inout.load_ply(model_path, vertex_scale=self.scale_to_meter)
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(model["pts"])
            models.append(model)

        mmcv.dump(models, cache_path, protocol=4)
        return models


def get_custom_rgbd_metadata(obj_names, ref_key):
    data_ref = getattr(ref, ref_key)
    loaded_models_info = data_ref.get_models_info() if hasattr(data_ref, "get_models_info") else {}
    cur_sym_infos = {}
    for i, obj_name in enumerate(obj_names):
        obj_id = data_ref.obj2id.get(obj_name, None)
        sym_info = None
        if obj_id is not None and str(obj_id) in loaded_models_info:
            model_info = loaded_models_info[str(obj_id)]
            if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
                sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
                sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
        cur_sym_infos[i] = sym_info
    return {"thing_classes": obj_names, "sym_infos": cur_sym_infos}


SPLITS_CUSTOM_RGBD = {}


def register_with_name_cfg(name, data_cfg=None):
    logger.info("register dataset: %s", name)
    if name in SPLITS_CUSTOM_RGBD:
        used_cfg = SPLITS_CUSTOM_RGBD[name]
    else:
        if data_cfg is None:
            raise ValueError(f"dataset name {name} is not registered")
        used_cfg = data_cfg

    dataset = CUSTOM_RGBD_Dataset(used_cfg)
    DatasetCatalog.register(name, dataset)
    MetadataCatalog.get(name).set(
        id=used_cfg.get("id", name),
        ref_key=used_cfg.get("ref_key", "custom_data_1"),
        objs=dataset.objs,
        eval_error_types=used_cfg.get("eval_error_types", ["ad", "rete", "proj"]),
        evaluator_type=used_cfg.get("evaluator_type", "bop"),
        **get_custom_rgbd_metadata(obj_names=dataset.objs, ref_key=used_cfg.get("ref_key", "custom_data_1")),
    )


def get_available_datasets():
    return list(SPLITS_CUSTOM_RGBD.keys())