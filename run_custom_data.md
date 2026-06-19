# Running GDRNPP on Your Own Images

Practical, staged guide for running GDRNPP inference on custom RGB-D data (the
RealSense `cup`, later KITchen objects), validated against the actual state of
this checkout. Read the **honest constraint** first — it determines what "running
on the cup" can and cannot mean.

---

## 0. The honest constraint (read this first)

**GDRNPP is instance-level, not model-free.** Unlike GigaPose (which matches
templates at test time), a GDRNPP pose network is trained for a *specific set of
objects* and predicts dense 2D→3D correspondences in each object's own coordinate
frame. Consequences:

- There are **no trained weights for the cup** (only the 7 BOP datasets are
  present under `output/GDRNPP/`). Running any pretrained model on the cup will
  **not** produce a meaningful pose.
- A real cup pose requires **training on the cup first** (the KITchen path, §6).
- So "cup to test now" means a **plumbing test**: prove the install + custom-data
  loader run end-to-end and emit a results CSV, accepting the pose is garbage
  until training.

To validate the *install itself* with a genuinely correct pose, run a pretrained
BOP model on its own data first (§3). De-risk the install (§3), then the data
path (§4), then train (§6).

---

## 1. State of this checkout (already done)

The custom-data scaffolding is **already wired** (from the "custom data" commit):

| Piece | Path | Purpose |
|---|---|---|
| Custom loader | `core/gdrn_modeling/datasets/custom_rgbd.py` | Scans `rgb/`, finds `depth/`, reads `camera.json`, loads CAD; registered in `dataset_factory.py` (`custom_rgbd`) |
| Object registry | `ref/custom_data_1.py` | Reads object names from `datasets/custom_data_1/objects.txt`, CADs from `datasets/custom_data_1/models/` |
| Data cfg | `configs/custom_data_1/custom_data_1.json` | `mod_name=custom_rgbd`, RGB-D, `scale_to_meter=0.001` |
| Experiment cfg | `configs/gdrn/custom_data_1/…custom_data_1.py` | `GDRN_double_mask`, `LOAD_DETS_TEST=False`, depth on |
| Pretrained weights | `output/GDRNPP/gdrn/{lmo,tudl,ycbv,tless,icbin,hb,itodd}/…` | BOP objects only (69 GB total) |
| YOLOX detectors | `output/GDRNPP/yolox/bop_pbr/…` | Per-BOP-dataset detection |

**Still missing:** a runnable environment, any BOP dataset on disk
(`datasets/BOP_DATASETS` does not exist), `test_bboxes`, the staged cup data, and
cup weights.

### Hardware / runtime facts
- GPU: **RTX A2000 12 GB**; host CUDA toolkit **11.5**; nvidia driver present.
- **No `gdrnpp` conda env**, user **not in `docker` group**, **no passwordless
  sudo**. Docker client v29.5.2 is installed.
- Decision: **use Docker** (intended runtime, CUDA 11.6 image — avoids fighting
  the host CUDA 11.5), driven via `sudo` using the `!` prefix in the session.

---

## 2. Build the Docker image (one-time, ~20–40 min)

A `.dockerignore` excludes the 69 GB `output/`, `datasets/`, `.git`, etc. from the
build context (they are bind-mounted at runtime instead). The extensions compile
**in-place inside the repo tree** (`build_ext --inplace`), so the image bakes the
code + compiled `.so`; do **not** bind-mount over `core/csrc` or `lib/egl_renderer`
at runtime or you shadow them.

```bash
# run in-session with `!` so the sudo password prompt reaches your terminal
cd /home/pose/dipl/gdrnpp_bop2022
sudo docker build -t gdrnpp:cuda11.6 -f docker/Dockerfile .
```

- Build needs **no GPU**. If it fails on a cpp extension or a pip pin, patch
  `docker/Dockerfile` / `scripts/{install_deps,compile_all}.sh` and rebuild.
- Running later needs **`nvidia-container-toolkit`** on the host for `--gpus all`.

### 2.1 Environment fixes baked into the build (hard-won — do not regress)

The repo's stock `docker/Dockerfile` + `requirements.txt` produced a **silently
broken** env. The working recipe is **torch 1.13.1 / torchvision 0.14.1 / CUDA
11.6 / Python 3.8 / pip <24.1**. Every fix below is now in
`docker/Dockerfile`, `requirements/requirements.txt`,
`requirements/constraints.txt`, and `scripts/install_deps.sh`:

| Symptom | Root cause | Fix |
|---|---|---|
| `torch.cuda.is_available()=False`, `version.cuda=None` | unpinned `conda install pytorch` grabbed CPU-only 2.4.x (no cu116 build) | pin `pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6` |
| `numpy` ABI warning (`API 0xf vs 0xd`) | numpy 2.x vs torch 1.13 | pin `numpy=1.23` |
| `ResolutionImpossible` / `Ignoring ... invalid metadata: torch (>=1.9.*)` | **pip ≥24.1** rejects PL 1.x's malformed PEP 440 metadata → whole install aborts | **`pip install "pip<24.1"`** before requirements (keystone fix) |
| `No matching distribution for meshplex` | meshplex needs py≥3.9; aborts the whole `-r` install | commented out (only used by the unused vispy renderer) |
| `loguru`/`setproctitle`/etc. missing at runtime | `install_deps.sh` had no `set -e`, swallowing the aborted install | made the requirements step fail loudly (`|| exit 1`) + a build-time import guard |
| `ImportError: _compare_version from torchmetrics` | torchmetrics ≥0.11 (too new for PL 1.7.7) | pin `torchmetrics==0.10.3` |
| `ModuleNotFoundError: pytorch_lightning.lite.wrappers` | PL 2.x removed LightningLite (repo's `GDRN_Lite(LightningLite)`) | pin `pytorch-lightning==1.7.7` |
| `detectron2 _C.so: undefined symbol: ...torch::Library...` | detectron2 built against a torch that later got swapped (a manual `pip install` pulled torch 2.4.1) | `PIP_CONSTRAINT=requirements/constraints.txt` locks torch for **all** pip (build + container) |
| `module 'PIL.Image' has no attribute 'LINEAR'` | Pillow ≥10 removed the alias detectron2 0.6 uses | pin `Pillow==9.5.0` (last step) |
| `assimp library not found` (pyassimp) | `libassimp` apt-install in `install_deps.sh` ran after apt lists were cleared → silently failed | dedicated `apt-get install libassimp-dev` in the Dockerfile |
| missing `cv2` / `detectron2` | opencv not in requirements; detectron2 commented out | added `opencv-python`; build detectron2 `@v0.6` from source; `mmcv-full==1.7.1` via openmmlab cu116/torch1.13 wheel |
| install pulls torch 2.x via `torchvision`/`pytorch3d`/`pytorch-lightning` (unpinned) | bare entries resolve to latest → drag torch forward | commented `torchvision`/`pytorch3d` (conda/unused) in requirements |
| dropped, unused & conflict-prone | `onnx`/`onnxruntime`/`onnx-simplifier`/`pyro-ppl`/`deepspeed` — 0 imports, ancient pins deadlock the resolver | commented out |

**Key safety nets now in the image:**
- `requirements/constraints.txt` (fed via `PIP_CONSTRAINT`) pins
  `torch / torchvision / pytorch-lightning / torchmetrics` for **every** pip call,
  including inside the running container — so a stray `pip install <x>` can no
  longer silently swap torch and break detectron2's compiled `_C`.
- A **build-time guard** RUN imports the critical deps and asserts torch stayed
  1.13.1; the build fails loudly if the env is incomplete (no more discovering a
  broken env at runtime).
- **Never `pip install` by hand in the container** to "fix" a missing module — it
  means the image is wrong; fix the Dockerfile/requirements and rebuild.

**Build cost note:** editing any file in the build context (incl. the Dockerfile,
since it's not in `.dockerignore`) busts the `COPY .` layer → re-runs
`install_deps` + detectron2 (source) + the compiles (~15 min). The conda+torch
layer (~340 MB download) is *before* `COPY`, so it stays cached. Don't use
`--no-cache` (re-downloads torch every time; prone to `IncompleteRead`).

### Standard run command (mounts data, keeps baked code+extensions)
```bash
sudo docker run --gpus all -it --rm --shm-size=16g \
  -v /home/pose/dipl/gdrnpp_bop2022/output:/workspace/output \
  -v /home/pose/dipl/gdrnpp_bop2022/datasets:/workspace/datasets \
  gdrnpp:cuda11.6
```
**`--shm-size=16g` is required** — the default 64 MB `/dev/shm` makes PyTorch
DataLoader workers crash with `Bus error ... out of shared memory`. (Alternatively
`--ipc=host`.)
For live editing of the (non-compiled) custom-data code, also mount the specific
dirs that contain **no** `.so` — safe because the extensions live only under
`core/csrc` and `lib/egl_renderer`:
```bash
  -v /home/pose/dipl/gdrnpp_bop2022/configs:/workspace/configs \
  -v /home/pose/dipl/gdrnpp_bop2022/ref:/workspace/ref \
  -v /home/pose/dipl/gdrnpp_bop2022/core/gdrn_modeling/datasets:/workspace/core/gdrn_modeling/datasets \
```

---

## 3. Smoke tests (validate the install)

### 3a. Import / build check (no data needed)
Inside the container:
```bash
conda activate gdrnpp   # entrypoint does this automatically
# env (expect: 1.13.1 11.6 True):
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# python deps in one shot:
python -c "import detectron2,cv2,mmcv,timm,transforms3d,albumentations,open3d; print('deps ok', detectron2.__version__)"
# the heavy native import (exercises OpenGL/EGL + pyassimp+libassimp + cv2 + torch):
python -c "from lib.egl_renderer.egl_renderer_v3 import EGLRenderer; print('egl ok')"
# compiled CUDA extensions (real module names):
python -c "import core.csrc.ransac_voting.ransac_voting; print('ransac ok')"
# list what actually built:
find core/csrc lib/egl_renderer -name "*.so"
```
Expected `.so`: `CppEGLRenderer` (egl), `ransac_voting`, `flow_cuda`,
`torch_nndistance_aten` (chamfer), `uncertainty_pnp` (`ext`), and `fps`.
If an extension import fails or its `.so` is missing, recompile inside the
container: `bash scripts/compile_all.sh` (note its `fps` step runs a bare
`python setup.py` with no build command, and the script has no `set -e`, so a
single failure is silent — check each `.so` exists).

### 3b. BOP real-pose test (LM-O) — DONE ✅ (validated 2026-06-18)

Goal: genuinely correct poses, proving the full pipeline (detectron2 + extensions
+ pose head + BOP eval). **Result on this box** (RGB-only, no depth refine, lmo
pretrained weights): MSPD 84.98, MSSD 63.84, ADD-0.1d 49.7%, reS_10 79.8%,
teS_10 90.9% over the 8 LM-O objects — strong rotation/projection accuracy =
install confirmed. (ADD is modest because this is RGB-only; depth refinement —
`test_gdrn_depth_refine.sh` — boosts it.)

**Exact wiring used** (data lives at `~/dipl/datasets/lm-o`, mounted into the
container at the path the config expects):

1. LM-O data: `models/`, `models_eval/`, `test/` (scene 000002), plus the
   `GDRNPP_bop22_test_bboxes` repo — all under `~/dipl/datasets/lm-o`.
2. **git-lfs**: the bboxes repo clones as LFS *pointers* (132-byte files). Pull
   the real content:
   ```bash
   sudo apt-get install -y git-lfs
   cd ~/dipl/datasets/lm-o/GDRNPP_bop22_test_bboxes && git lfs install && \
     git lfs pull --include="datasets/BOP_DATASETS/lmo/**"
   ```
   Copy the real json to `~/dipl/datasets/lm-o/test/test_bboxes/yolox_x_640_lmo_pbr_lmo_bop_test.json`.
3. **`train_pbr` stub**: the lmo config registers `lmo_pbr_train` even for eval,
   and `LM_PBR_Dataset.__init__` only asserts the dir exists (eval never loads its
   images). `mkdir ~/dipl/datasets/lm-o/train_pbr` is enough.
4. **`test_targets_bop19.json`**: not in the BOP test archive — generate it from
   `scene_gt.json` for exactly the images in the bbox json (keys are `"scene/im"`),
   emitting `{im_id, inst_count, obj_id, scene_id}` rows. Write to
   `~/dipl/datasets/lm-o/test_targets_bop19.json`.
5. **Eval renderer**: the config defaults to `ERROR_TYPES=...,vsd,...` +
   `RENDERER_TYPE="cpp"`, but `bop_renderer` is a dangling symlink → eval
   subprocess fails → `error:vsd_... does not exist`. Either drop `vsd`
   (`ERROR_TYPES="mspd,mssd,ad,reS,teS"`, no renderer needed) or set
   `RENDERER_TYPE="egl"`. Only VSD needs a renderer.
6. Run (note `--shm-size` and the nested lmo mount):
   ```bash
   sudo docker run --gpus all -it --rm --shm-size=16g \
     -v /home/pose/dipl/gdrnpp_bop2022/output:/workspace/output \
     -v /home/pose/dipl/gdrnpp_bop2022/datasets:/workspace/datasets \
     -v /home/pose/dipl/datasets/lm-o:/workspace/datasets/BOP_DATASETS/lmo \
     gdrnpp:cuda11.6
   # inside:
   ./core/gdrn_modeling/test_gdrn.sh \
     configs/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo.py \
     0 output/GDRNPP/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo/model_final_wo_optim.pth
   ```

**Extra env pin found at runtime** (now in `constraints.txt`): `yapf==0.32.0`
(mmcv 1.7.1 calls `FormatCode(..., verify=True)`; yapf ≥0.40 removed `verify`).

---

## 4. Cup plumbing test (custom-data path)

Proves data registration + image/depth/camera/CAD loading + detection feeding +
forward pass + overlay. **Done ✅ 2026-06-19** (cup-as-ape proxy). Pose is rough —
no cup weights — but localization tracks the cup; orientation is the proxy object's.

**The checkpoint problem & the fix.** GDRNPP needs a checkpoint, and there is none
for the cup. A class-aware checkpoint (e.g. 8-class lmo) won't load against a
1-object custom config (head size mismatch). So use a **single-object (SO)
checkpoint** — class-agnostic, 1-class → loads cleanly into a 1-object config. Pick
an SO object as a stand-in and use **that object's CAD + train metadata** (the model
predicts in the object's own frame). We used lmo **ape** (its CAD ~76×78×92 mm is
close to the cup's ~90 mm). NB: the lmo **can** SO checkpoint was a **truncated
download** (386 MB vs 410 MB, `zipfile.is_zipfile()==False`) — verify integrity
before trusting any zoo file.

### 4a. Stage the data (`tools/`-style one-off script)
Layout under `datasets/custom_data_1/` (mounted into the container): `rgb/`,
`depth/`, `camera.json` (`{"cam_K":[9], "depth_scale":..}`), `models/obj_000001.ply`
(+ `models_info.json` key `"1"`), `objects.txt` (one name → id 1).
- Cup RGB-D source: `…/FoundationPose/demo_data/realsense_cup/{rgb,depth,masks,camera.json}`.
- **Stage only frames that have a mask** and **renumber them `000000..`** to match
  `custom_rgbd`'s enumeration keying (`scene_im_id = "scene/<enum-index>"`).
- `obj_000001.ply` = the **proxy object's** CAD (we copied lmo `obj_000001.ply` =
  ape) + `models_info["1"]` = that object's info. BOP mm → `scale_to_meter=0.001`.

### 4b. Detections (reuse the masks you already have)
Convert the per-frame masks to GDRNPP `test_bboxes` JSON: keyed `"scene/<enum-idx>"`
→ `[{"bbox_est":[x,y,w,h], "obj_id":1, "score":1.0, "time":0.0}]` (bbox from
`np.where(mask>0)`). Set `MODEL.LOAD_DETS_TEST=True` and
`DATASETS.DET_FILES_TEST=("…/cup_detections.json",)`. Keying must match the staged
frame indices exactly, or collate `KeyError`s.

### 4c. Config + run
`configs/custom_data_1/cup.json` (custom_rgbd data cfg: objs, ref_key, h=360/w=640,
`with_depth=false`, `use_cache=false`) and `configs/gdrn/cup_as_can/cup.py`
(`_base_` = the SO `ape.py` so the checkpoint matches; `TRAIN=("lmo_ape_train_pbr",)`
for the renderer's CAD metadata; `TEST=("custom_cup",)`; **`TEST.SAVE_RESULTS_ONLY=True`
+ `VAL.SAVE_BOP_CSV_ONLY=True`** since there's no GT → skip BOP eval).

Run with the **SO ape** checkpoint (mount `configs/`, the dataset loaders, `ref/`,
plus the lmo data for the ape CAD; `--shm-size=16g`):
```bash
sudo docker run --gpus all -it --rm --shm-size=16g \
  -v $PWD/output:/workspace/output -v $PWD/datasets:/workspace/datasets \
  -v /home/pose/dipl/datasets/lm-o:/workspace/datasets/BOP_DATASETS/lmo \
  -v $PWD/configs:/workspace/configs \
  -v $PWD/core/gdrn_modeling/datasets:/workspace/core/gdrn_modeling/datasets \
  -v $PWD/ref:/workspace/ref  gdrnpp:cuda11.6
# inside (yapf not baked yet -> reinstall once):
pip install yapf==0.32.0
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/cup_as_can/cup.py 0 \
  output/GDRNPP/gdrn/lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/ape/model_final_wo_optim.pth
```
Output: `output/gdrn/cup_as_can/ape_on_cup/inference_*/custom_cup/results.pkl`
(dict `"scene/im"` → `[{R(3×3), t(3,) in **meters**, obj_id, mask…}]`).

### 4d. Visualize (FoundationPose-style overlay)
`tools/vis_gdrn_custom.py` overlays a posed 3D box + XYZ axes (reusable for any
custom_rgbd run). `t` is in **meters**; box corners come from `models_info` (mm→m).
Run in an env with numpy+opencv (e.g. `gigapose`); write to a **host-writable** dir
(`output/` is root-owned by the container):
```bash
conda activate gigapose
python tools/vis_gdrn_custom.py \
  --results output/gdrn/cup_as_can/ape_on_cup/inference_model_final_wo_optim/custom_cup/results.pkl \
  --rgb_dir datasets/custom_data_1/rgb --camera datasets/custom_data_1/camera.json \
  --models_info datasets/custom_data_1/models/models_info.json \
  --out datasets/custom_data_1/vis --video
```

---

## 4e. Open follow-ups (next steps)

- [ ] **Bake the image** so runtime patches are permanent. `yapf==0.32.0` is in
  `constraints.txt` but the live image still needs the in-container `pip install
  yapf==0.32.0` each run. One consolidated rebuild fixes it:
  `cd /home/pose/dipl/gdrnpp_bop2022 && sudo docker build -t gdrnpp:cuda11.6 -f docker/Dockerfile .`
  (also bakes the lmo config `vsd`→drop + `RENDERER_TYPE=egl` edit). After this the
  documented run commands work with zero manual steps.
- [ ] **Real cup poses require training** — the cup-as-ape run (§4) is plumbing only;
  there are no cup weights. Train on the cup/KITchen (see `train_model.md` / §6),
  then run inference. The cup data is **RGB-D**, so the accuracy payoff is training
  **+ `test_gdrn_depth_refine.sh`** (depth refinement is what lifts ADD from the
  ~50% RGB-only level we saw on LM-O into the published range).

---

## 5. Quality reality check

A running pipeline ≠ good poses. The cup masks/detections from the RealSense run
were tiny/far in some frames; combined with **no cup-trained weights**, §4 cannot
yield good poses. Suspect the **inputs and the missing training**, not the model.

---

## 6. The real fix: train on KITchen (later)

See [train_model.md](train_model.md) for the full KITchen training path
(dataset registration, config, loader sanity check, `train_gdrn.sh`,
`test_gdrn.sh`). Key reminders for a first run:
- Start with **one object** (the cup), not all 111.
- Confirm mesh scale (mm) matches the pose-annotation units.
- Use `GDRN_double_mask` + ConvNeXt + `AMODAL_CLIP`, keep strong domain
  randomization for real data.
- Verify depth scale before enabling depth-based options.

---

## 7. Debugging order that works

1. **Env**: `torch.cuda.is_available()`, detectron2, each compiled extension import.
2. **Install proof**: §3b LM-O real-pose test produces correct poses.
3. **Units**: `mesh.extents` in mm; intrinsics match image resolution.
4. **Custom data path**: §4 — registration → load → forward → CSV.
5. **Sanity-check poses**: `det(R)≈1`, translation plausible (mm).
6. **Overlay**: project CAD vertices with `K @ (R@V + t)` onto RGB; eyeball it.
7. **Train** (§6) once the plumbing is proven.

## 8. Known gotchas

- `output/` is **69 GB** — never put it in the Docker build context; bind-mount it.
- Compiled extensions are **in-place** in the repo tree → don't bind-mount over
  `core/csrc` or `lib/egl_renderer`.
- Host CUDA is **11.5**; the image is **11.6** — build/run inside the container,
  not on the host.
- `cup` has **no trained weights** — §4 is plumbing only; real poses need §6.
- The `bop_renderer` repo-root entry is a symlink to a foreign path
  (`/data/lxy/bop_renderer`) and is excluded from the build context.
