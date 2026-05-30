import hashlib
import logging
import os.path as osp
import sys
import time
from collections import OrderedDict

import mmcv
import numpy as np
from transforms3d.quaternions import mat2quat

import ref
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from lib.pysixd import inout, misc
from lib.utils.mask_utils import binary_mask_to_rle
from lib.utils.utils import dprint, lazy_property


logger = logging.getLogger(__name__)
PROJ_ROOT = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), "../../.."))
DEFAULT_DATASET_ROOT = osp.join(PROJ_ROOT, "datasets", "KITchen")


class KITCHEN_PBR_Dataset:
    """KITchen data loader for a serialized manifest.

    The manifest can be a list of Detectron2-style records or a dict with a
    `dataset_dicts` / `samples` list. This keeps the loader small and makes the
    KITchen conversion step explicit.
    """

    def __init__(self, data_cfg):
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.dataset_root = data_cfg.get("dataset_root", DEFAULT_DATASET_ROOT)
        self.models_root = data_cfg.get("models_root", osp.join(self.dataset_root, "models", "models"))
        self.ann_file = data_cfg["ann_file"]

        self.objs = data_cfg.get("objs", [])
        if not self.objs and hasattr(ref, "kitchen"):
            self.objs = getattr(ref.kitchen, "objects", [])

        self.scale_to_meter = data_cfg.get("scale_to_meter", 0.001)
        self.with_masks = data_cfg.get("with_masks", True)
        self.with_depth = data_cfg.get("with_depth", True)
        self.height = data_cfg.get("height", 480)
        self.width = data_cfg.get("width", 640)
        self.cache_dir = data_cfg.get("cache_dir", osp.join(PROJ_ROOT, ".cache"))
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg.get("num_to_load", -1)
        self.filter_invalid = data_cfg.get("filter_invalid", True)
        self.ref_key = data_cfg.get("ref_key", "kitchen")

        if not self.objs:
            self.objs = self._infer_objects_from_manifest()

        self.obj2label = OrderedDict((obj_name, idx) for idx, obj_name in enumerate(self.objs))

    def _infer_objects_from_manifest(self):
        manifest = mmcv.load(self.ann_file)
        if isinstance(manifest, dict):
            if "dataset_dicts" in manifest:
                dataset_dicts = manifest["dataset_dicts"]
            elif "samples" in manifest:
                dataset_dicts = manifest["samples"]
            else:
                dataset_dicts = []
        elif isinstance(manifest, list):
            dataset_dicts = manifest
        else:
            dataset_dicts = []

        inferred = []
        seen = set()
        for record in dataset_dicts:
            annos = record.get("annotations") or record.get("annos") or []
            for anno in annos:
                obj_name = anno.get("obj_name") or anno.get("category_name")
                if obj_name is None and "obj_id" in anno and hasattr(ref, "kitchen"):
                    obj_name = ref.kitchen.id2obj.get(int(anno["obj_id"]), None)
                if obj_name is None or obj_name in seen:
                    continue
                seen.add(obj_name)
                inferred.append(obj_name)
        return inferred

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
            logger.info("load cached dataset dicts from {}".format(cache_path))
            return mmcv.load(cache_path)

        t_start = time.perf_counter()
        logger.info("loading dataset dicts: {}".format(self.name))

        manifest = mmcv.load(self.ann_file)
        if isinstance(manifest, dict):
            if "dataset_dicts" in manifest:
                dataset_dicts = manifest["dataset_dicts"]
            elif "samples" in manifest:
                dataset_dicts = manifest["samples"]
            else:
                raise ValueError("KITchen manifest dict must contain `dataset_dicts` or `samples`.")
        elif isinstance(manifest, list):
            dataset_dicts = manifest
        else:
            raise ValueError("Unsupported KITchen manifest type: {}".format(type(manifest)))

        normalized_dicts = []
        for record in dataset_dicts:
            normalized = dict(record)
            file_name = normalized.get("file_name") or normalized.get("rgb_file") or normalized.get("rgb_path")
            if file_name is None:
                raise ValueError("KITchen record is missing an RGB image path: {}".format(record))
            if not osp.isabs(file_name):
                file_name = osp.join(self.dataset_root, file_name)
            normalized["file_name"] = osp.relpath(file_name, PROJ_ROOT)

            if self.with_depth:
                depth_file = normalized.get("depth_file") or normalized.get("depth_path")
                if depth_file is not None and not osp.isabs(depth_file):
                    depth_file = osp.join(self.dataset_root, depth_file)
                if depth_file is not None:
                    normalized["depth_file"] = osp.relpath(depth_file, PROJ_ROOT)

            normalized.setdefault("height", self.height)
            normalized.setdefault("width", self.width)

            cam = normalized.get("cam") or normalized.get("K") or normalized.get("camera_matrix")
            if cam is not None:
                normalized["cam"] = np.array(cam, dtype=np.float32).reshape(3, 3)

            annos = normalized.get("annotations") or normalized.get("annos") or []
            normalized_annos = []
            for anno in annos:
                obj_name = anno.get("obj_name") or anno.get("category_name")
                if obj_name is None and "obj_id" in anno and hasattr(ref, "kitchen"):
                    obj_name = ref.kitchen.id2obj.get(int(anno["obj_id"]), None)

                if self.objs and obj_name not in self.objs:
                    continue

                if obj_name is None:
                    raise ValueError("KITchen annotation is missing object name/id: {}".format(anno))

                cur_label = self.obj2label.get(obj_name)
                if cur_label is None:
                    if not self.objs:
                        self.obj2label[obj_name] = len(self.obj2label)
                        cur_label = self.obj2label[obj_name]
                    else:
                        continue

                inst = dict(anno)
                inst["category_id"] = cur_label
                inst.setdefault("bbox_mode", BoxMode.XYWH_ABS)

                if "pose" in inst:
                    pose = np.array(inst["pose"], dtype=np.float32).reshape(3, 4)
                elif "R" in inst and "t" in inst:
                    R = np.array(inst["R"], dtype=np.float32).reshape(3, 3)
                    t = np.array(inst["t"], dtype=np.float32).reshape(3)
                    pose = np.hstack([R, t.reshape(3, 1)])
                else:
                    pose = None
                if pose is not None:
                    inst["pose"] = pose
                    inst["quat"] = mat2quat(pose[:, :3]).astype("float32")
                    inst["trans"] = pose[:, 3].astype("float32")

                if "bbox_obj" not in inst and "bbox" in inst:
                    inst["bbox_obj"] = inst["bbox"]

                if self.with_masks and "segmentation" not in inst:
                    mask_file = inst.get("mask_file") or inst.get("mask_path")
                    if mask_file is not None and not osp.isabs(mask_file):
                        mask_file = osp.join(self.dataset_root, mask_file)
                    if mask_file is not None and osp.exists(mask_file):
                        mask_single = mmcv.imread(mask_file, "unchanged")
                        if mask_single.ndim == 3:
                            mask_single = mask_single[:, :, 0]
                        inst["segmentation"] = binary_mask_to_rle(mask_single.astype("uint8"), compressed=True)

                normalized_annos.append(inst)

            if len(normalized_annos) == 0:
                continue

            normalized["annotations"] = normalized_annos
            normalized["dataset_name"] = self.name
            normalized_dicts.append(normalized)

        if self.num_to_load > 0:
            self.num_to_load = min(int(self.num_to_load), len(normalized_dicts))
            normalized_dicts = normalized_dicts[: self.num_to_load]

        logger.info("loaded {} dataset dicts, using {}s".format(len(normalized_dicts), time.perf_counter() - t_start))
        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(normalized_dicts, cache_path, protocol=4)
        logger.info("Dumped dataset_dicts to {}".format(cache_path))
        return normalized_dicts

    @lazy_property
    def models_info(self):
        models_info_path = osp.join(self.models_root, "models_info.json")
        if not osp.exists(models_info_path):
            return {}
        return mmcv.load(models_info_path)

    @lazy_property
    def models(self):
        cache_path = osp.join(self.models_root, "models_{}.pkl".format("_".join(self.objs) if self.objs else "all"))
        if osp.exists(cache_path) and self.use_cache:
            return mmcv.load(cache_path)

        models = []
        for obj_name in self.objs:
            obj_id = ref.kitchen.obj2id.get(obj_name, None) if hasattr(ref, "kitchen") else None
            if obj_id is None:
                raise KeyError("Unknown KITchen object name: {}".format(obj_name))
            model_path_candidates = [
                osp.join(self.models_root, f"obj_{obj_id:06d}.ply"),
                osp.join(self.models_root, f"{obj_name}.ply"),
            ]
            model_path = next((path for path in model_path_candidates if osp.exists(path)), None)
            if model_path is None:
                raise FileNotFoundError("No model file found for KITchen object {}".format(obj_name))
            model = inout.load_ply(model_path, vertex_scale=self.scale_to_meter)
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(model["pts"])
            models.append(model)

        mmcv.dump(models, cache_path, protocol=4)
        return models

    def __len__(self):
        return self.num_to_load

    def image_aspect_ratio(self):
        return self.width / self.height


def get_kitchen_metadata(obj_names, ref_key):
    data_ref = ref.__dict__[ref_key]
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


SPLITS_KITCHEN_PBR = {}


def register_with_name_cfg(name, data_cfg=None):
    dprint("register dataset: {}".format(name))
    if name in SPLITS_KITCHEN_PBR:
        used_cfg = SPLITS_KITCHEN_PBR[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg

    dataset = KITCHEN_PBR_Dataset(used_cfg)
    DatasetCatalog.register(name, dataset)
    MetadataCatalog.get(name).set(
        id="kitchen",
        ref_key=used_cfg.get("ref_key", "kitchen"),
        objs=dataset.objs,
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        **get_kitchen_metadata(obj_names=dataset.objs, ref_key=used_cfg.get("ref_key", "kitchen")),
    )


def get_available_datasets():
    return list(SPLITS_KITCHEN_PBR.keys())
