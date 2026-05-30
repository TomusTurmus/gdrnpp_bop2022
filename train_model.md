# Retraining GDRNPP On KITchen

This repo does not know about KITchen out of the box, so retraining is a two-part job:

1. Make KITchen look like a dataset this code can read.
2. Point a GDRNPP config at that dataset and launch training with the repo script.

The repo’s pose training entry point is [core/gdrn_modeling/train_gdrn.sh](core/gdrn_modeling/train_gdrn.sh), which calls [core/gdrn_modeling/main_gdrn.py](core/gdrn_modeling/main_gdrn.py). Dataset registration happens through the dataset factory in [core/gdrn_modeling/datasets/dataset_factory.py](core/gdrn_modeling/datasets/dataset_factory.py).

For multi-GPU runs, `train_gdrn.sh` counts GPUs from the comma-separated device list you pass as the second argument, then forwards that count to `main_gdrn.py`. `main_gdrn.py` switches to DDP automatically when `--num-gpus` is greater than 1.

Because the repo is designed to run inside the Docker image from [docker/Dockerfile](docker/Dockerfile), the KITchen paths in the config should resolve under `/workspace/datasets/KITchen` inside the container. The simplest setup is to mount or copy your extracted KITchen folder to that location before training.

## What The Code Expects

The training code expects datasets under a `datasets/` folder at the repo root, usually in BOP-style layout such as `datasets/BOP_DATASETS/<dataset_name>/...`.

If you want to keep KITchen in your existing folder, the better choice is usually to adapt the dataset config and loader to point there directly instead of moving the archive into the repo. That keeps your original data location intact and avoids duplicate copies.

If KITchen is already close to BOP layout, a one-time BOP-style restructuring is still the cleanest long-term option because the existing loaders already know how to read that format. If it is not close to BOP, keep the data where it is and write a small custom loader/config that reads from that path.

Given your current KITchen layout, the custom config/loader route is the better fit. You already have the main ingredients the model needs, but they are organized as separate folders instead of a BOP scene tree:

* 2D bounding boxes
* depth datasets, three sets
* RGB datasets, three sets
* masks
* models
* object names

That means you can map the dataset directly into the fields the repo expects without first reshaping the whole archive into BOP directories. The 2D boxes can drive the detector side, while RGB, depth, masks, and models can feed the pose training loader.

The most likely split is to treat the three RGB/depth groups as separate train/val/test or train/train2/test partitions, depending on how KITchen defines them.

If you prefer, you can also make your own 80/10/10 split from the available images. That is fine for creating a custom train/validation/test setup, but it does not replace missing supervision. With the data you have right now, the `labels/` folder is only giving 2D bounding boxes, not 6D pose annotations.

For KITchen, the safest approach is to unpack the archive into something like:

```text
datasets/BOP_DATASETS/kitchen/
```

If you want to keep the data outside the repo, create a symlink into that location instead.

The training dataloader needs, at minimum:

* RGB images
* object masks or visible masks
* 6D pose annotations
* camera intrinsics
* object meshes or models

KITchen is an RGB-D 6D pose dataset, so it is a good fit if your extracted archive includes those items.

## Recommended Workflow

### 1. Unzip And Inspect KITchen

Unpack the archive in `~/dipl/datasets/KITchen` and verify the internal structure.

You want to confirm these things before touching the code:

* How images are stored.
* Whether depth is available and how it is named.
* Whether masks are already provided.
* Whether poses are stored per image or per scene.
* Whether object meshes are included.

If KITchen already provides BOP-like files such as `scene_gt.json`, `scene_gt_info.json`, and `scene_camera.json`, the conversion work is much smaller.

### 2. Convert KITchen To A Format The Loader Can Read

GDRNPP’s dataset loaders in this repo follow the same pattern as the existing BOP datasets. The loader classes in [core/gdrn_modeling/datasets/](core/gdrn_modeling/datasets/) read scene-level JSON files, RGB images, depth files, and masks.

If KITchen is not already in that format, convert it into a BOP-style tree. The usual target layout is:

```text
kitchen/
	000000/
		rgb/
		depth/
		mask/
		mask_visib/
		scene_gt.json
		scene_gt_info.json
		scene_camera.json
```

The important part is not the exact folder names, but that your custom loader can find the same information the existing loaders use.

### 3. Add A Dataset Registration File

The repo already supports custom datasets through the `DATA_CFG` path in [core/gdrn_modeling/datasets/dataset_factory.py](core/gdrn_modeling/datasets/dataset_factory.py).

If a dataset name is not one of the built-ins, the factory loads a config file from `cfg.DATA_CFG[name]`, expects a `mod_name`, and passes the rest of the config into that dataset module.

That means the clean way to add KITchen is:

* Copy the closest existing loader, usually one of the BOP-style train loaders such as `tless_pbr.py`, `ycbv_d2.py`, or `icbin_pbr.py`.
* Rename it to something like `kitchen_pbr.py` or `kitchen_real.py`.
* Update the paths and object list.
* Add the module name to the dataset factory’s module list.

The data config should include fields like:

* `mod_name`
* `name`
* `dataset_root`
* `models_root`
* `objs`
* `ref_key`
* `scale_to_meter`
* `with_masks`
* `with_depth`
* `height`
* `width`
* `cache_dir`
* `use_cache`
* `num_to_load`
* `filter_invalid`

### 4. Create A Training Config

Make a new config under `configs/gdrn/` for KITchen.

Start from the closest existing real-data config and change only what matters:

* `DATASETS.TRAIN` to your KITchen training split name.
* `DATASETS.TEST` if you want validation.
* `DATA_CFG` entries for your custom dataset.
* `OUTPUT_DIR` to a new experiment folder.
* `INPUT.WITH_DEPTH` if KITchen depth is part of training.
* `INPUT.MIN_SIZE_TRAIN` and `INPUT.MAX_SIZE_TRAIN` if your images are a different resolution.

If KITchen is real RGB-D data, do not start from a synthetic-only config unless you know you want that setup.

To stay close to the GDRNPP training recipe, keep these choices in mind when you build the KITchen config:

* Use ConvNeXt as the backbone rather than ResNet-34.
* Use the dual-mask pose model, typically `GDRN_double_mask`, so amodal and visible masks are predicted separately.
* Keep strong domain randomization in the augmentation pipeline if KITchen is mixed real-world data.
* Keep the bounding-box style aligned with the repo’s pose setup, usually `AMODAL_CLIP` for this family of configs.
* Tune learning rate, weight decay, and visible-threshold filtering from the KITchen validation split rather than copying numbers blindly from another dataset.
* If KITchen already provides visible and full masks, use them rather than collapsing everything into a single mask target.

Here is a practical starter skeleton for a KITchen config. Keep the folder names and split names aligned with your own archive layout:
I created a starter config at [configs/gdrn/kitchen/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_kitchen.py](configs/gdrn/kitchen/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_kitchen.py). It keeps the repo’s ConvNeXt + dual-mask setup and points `DATA_CFG` at JSON files in the same folder.

The matching loader and reference module are in [core/gdrn_modeling/datasets/kitchen_pbr.py](core/gdrn_modeling/datasets/kitchen_pbr.py) and [ref/kitchen.py](ref/kitchen.py). They expect a manifest file under `/home/pose/dipl/datasets/KITchen/manifests/` that already contains the image paths, poses, boxes, and masks in a serialized list of records.

I also created the first manifest files at [kitchen_train.pkl](/home/pose/dipl/datasets/KITchen/manifests/kitchen_train.pkl) and [kitchen_val.pkl](/home/pose/dipl/datasets/KITchen/manifests/kitchen_val.pkl). Those were generated from the labeled `000` split because that is the only split with matching 2D box annotations in the files you have right now.

For an explicit 80/10/10 setup, use [kitchen_train_80.pkl](/home/pose/dipl/datasets/KITchen/manifests/kitchen_train_80.pkl), [kitchen_val_10.pkl](/home/pose/dipl/datasets/KITchen/manifests/kitchen_val_10.pkl), and [kitchen_test_10.pkl](/home/pose/dipl/datasets/KITchen/manifests/kitchen_test_10.pkl).
```python
_base_ = ["../../../_base_/gdrn_base.py"]

OUTPUT_DIR = "output/gdrn/kitchen/convnext_AugCosyAAEGray_DMask_amodalClipBox_kitchen"

DATASETS = dict(
	TRAIN=("kitchen_train",),
	TRAIN2=(),
	TRAIN2_RATIO=0.0,
	TEST=("kitchen_val",),
	SYM_OBJS=[],
)

DATA_CFG = dict(
	kitchen_train=dict(
		mod_name="kitchen_pbr",
		name="kitchen_train",
		dataset_root="/home/pose/dipl/datasets/KITchen/<rgb_or_scene_train_root>",
		models_root="/home/pose/dipl/datasets/KITchen/models",
		objs=["<object_1>", "<object_2>"],
		ref_key="kitchen",
		scale_to_meter=1.0,
		with_masks=True,
		with_depth=True,
		height=480,
		width=640,
		cache_dir=".cache",
		use_cache=True,
		num_to_load=-1,
		filter_invalid=True,
	),
	kitchen_val=dict(
		mod_name="kitchen_pbr",
		name="kitchen_val",
		dataset_root="/home/pose/dipl/datasets/KITchen/<rgb_or_scene_val_root>",
		models_root="/home/pose/dipl/datasets/KITchen/models",
		objs=["<object_1>", "<object_2>"],
		ref_key="kitchen",
		scale_to_meter=1.0,
		with_masks=True,
		with_depth=True,
		height=480,
		width=640,
		cache_dir=".cache",
		use_cache=True,
		num_to_load=-1,
		filter_invalid=True,
	),
)

INPUT = dict(
	WITH_DEPTH=True,
	MIN_SIZE_TRAIN=480,
	MAX_SIZE_TRAIN=640,
	MIN_SIZE_TEST=480,
	MAX_SIZE_TEST=640,
)

SOLVER = dict(
	IMS_PER_BATCH=6,
	TOTAL_EPOCHS=160,
)

MODEL = dict(
	LOAD_DETS_TEST=True,
	BBOX_TYPE="AMODAL_CLIP",
	POSE_NET=dict(
		NAME="GDRN_double_mask",
		BACKBONE=dict(
			PRETRAINED="timm",
		),
	),
)
```

If you want the config to be runnable, the next step is to create the matching `kitchen_pbr` dataset loader module so the `mod_name` above resolves correctly.

### 5. Run A Small Loader Test First

Before full training, make sure the dataset registers and loads.

The existing loaders all follow the same pattern: register the dataset, inspect `DatasetCatalog`, and run a visual check on a few samples.

The cheapest validation is to load one batch or inspect a few rendered samples before launching a long training run.

### 6. Launch Training

From the repo root, the standard command is:

```bash
./core/gdrn_modeling/train_gdrn.sh <config_path> <gpu_ids> (other args)
```

Example:

```bash
./core/gdrn_modeling/train_gdrn.sh configs/gdrn/kitchen/my_kitchen_gdrn.py 0
```

For multiple GPUs, pass a comma-separated list such as `0,1`.

### 7. Evaluate The Checkpoint

After training, test with:

```bash
./core/gdrn_modeling/test_gdrn.sh <config_path> <gpu_ids> <ckpt_path> (other args)
```

If you only want to save predictions, the repo also has a results-only path in `save_gdrn.sh`.

## Important Repo-Specific Details

* The repo root is expected to be on `PYTHONPATH` by the launcher script.
* `train_gdrn.sh` automatically sets the GPU count from the comma-separated device list.
* Dataset registration is done before training starts, so bad paths fail early.
* If a dataset is not built into the repo, `cfg.DATA_CFG[name]` must point to a config file that includes `mod_name`.
* Existing loaders cache parsed annotations in `.cache`, which helps a lot on repeated runs.
* The default training path builds the renderer from the training dataset metadata unless you are evaluating only or `XYZ_ONLINE` is disabled.

## Practical Advice For KITchen

If this is your first retraining run, keep the first version simple:

* Use one object subset first instead of all 111 objects.
* Make sure training and evaluation splits are cleanly separated.
* Verify the object mesh scale matches the pose units used in the annotations.
* Confirm the depth scale before enabling depth-based options.
* Start with a smaller batch size if the images are large.

## Minimal Checklist

* Unzip KITchen into a dataset location visible to the repo.
* Confirm the data has RGB, poses, camera intrinsics, and meshes.
* Convert or adapt KITchen to the loader format used by the repo.
* Add a KITchen dataset config and registration module.
* Create a KITchen experiment config under `configs/gdrn/`.
* Run a loader sanity check.
* Train with `./core/gdrn_modeling/train_gdrn.sh`.
* Evaluate with `./core/gdrn_modeling/test_gdrn.sh`.

## Sources

* Repo training entry point: [core/gdrn_modeling/train_gdrn.sh](core/gdrn_modeling/train_gdrn.sh)
* Dataset registration: [core/gdrn_modeling/datasets/dataset_factory.py](core/gdrn_modeling/datasets/dataset_factory.py)
* Repo overview: [README.md](README.md)
* KITchen dataset page: https://abdelrahmanyounes.github.io/KITchen/
