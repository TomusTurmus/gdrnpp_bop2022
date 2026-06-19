#!/usr/bin/env python
"""Overlay GDRNPP custom-dataset pose predictions onto the RGB frames.

Mirrors the FoundationPose visualisation look (posed 3D bounding box + XYZ axes).
Reads the `results.pkl` produced by a SAVE_RESULTS_ONLY run of a custom_rgbd
dataset, and the BOP `models_info.json` for the object's 3D bounds.

GDRNPP results.pkl: dict keyed "scene/im" -> list of {obj_id, score, R(3x3),
t(3,) in METERS, ...}. RGB frames are named by the enumeration index used by
custom_rgbd (000000.png ...). Run in any env with numpy + opencv (e.g. gigapose).

Example:
  python tools/vis_gdrn_custom.py \
    --results output/gdrn/cup_as_can/ape_on_cup/inference_model_final_wo_optim/custom_cup/results.pkl \
    --rgb_dir datasets/custom_data_1/rgb \
    --camera datasets/custom_data_1/camera.json \
    --models_info datasets/custom_data_1/models/models_info.json \
    --out output/gdrn/cup_as_can/ape_on_cup/vis
"""
import os, json, pickle, argparse
import numpy as np
import cv2


def to_homo(pts):
    return np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)


def project_3d_to_2d(pt, K, ob_in_cam):
    pt = pt.reshape(4, 1)
    projected = K @ ((ob_in_cam @ pt)[:3, :])
    projected = projected.reshape(-1)
    projected = projected / projected[2]
    return projected.reshape(-1)[:2].round().astype(int)


def draw_xyz_axis(color, ob_in_cam, scale, K, thickness=3):
    axes = {(0, 0, 255): [1, 0, 0], (0, 255, 0): [0, 1, 0], (255, 0, 0): [0, 0, 1]}
    origin = tuple(project_3d_to_2d(np.array([0, 0, 0, 1.0]), K, ob_in_cam))
    out = color.copy()
    for bgr, axis in axes.items():
        tip = np.array([*[a * scale for a in axis], 1.0])
        end = tuple(project_3d_to_2d(tip, K, ob_in_cam))
        out = cv2.arrowedLine(out, origin, end, color=bgr, thickness=thickness,
                              line_type=cv2.LINE_AA, tipLength=0.15)
    return out


def draw_posed_3d_box(K, img, ob_in_cam, bbox, line_color=(0, 255, 0), linewidth=2):
    xmin, ymin, zmin = bbox.min(axis=0)
    xmax, ymax, zmax = bbox.max(axis=0)

    def line3d(start, end):
        pts = np.stack((start, end), axis=0).reshape(-1, 3)
        pts = (ob_in_cam @ to_homo(pts).T).T[:, :3]
        proj = (K @ pts.T).T
        uv = np.round(proj[:, :2] / proj[:, 2].reshape(-1, 1)).astype(int)
        return cv2.line(img, uv[0].tolist(), uv[1].tolist(), color=line_color,
                        thickness=linewidth, lineType=cv2.LINE_AA)

    for y in [ymin, ymax]:
        for z in [zmin, zmax]:
            img = line3d(np.array([xmin, y, z]), np.array([xmax, y, z]))
    for x in [xmin, xmax]:
        for z in [zmin, zmax]:
            img = line3d(np.array([x, ymin, z]), np.array([x, ymax, z]))
    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            img = line3d(np.array([x, y, zmin]), np.array([x, y, zmax]))
    return img


def bbox_from_models_info(models_info, obj_id):
    mi = models_info[str(obj_id)]
    mn = np.array([mi["min_x"], mi["min_y"], mi["min_z"]], dtype=float)
    sz = np.array([mi["size_x"], mi["size_y"], mi["size_z"]], dtype=float)
    corners = np.array([mn, mn + sz]) / 1000.0  # mm -> meters (GDRNPP t is in meters)
    return corners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--rgb_dir", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--models_info", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--axis_scale", type=float, default=0.05)
    ap.add_argument("--video", action="store_true", help="also write overlay.mp4")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = pickle.load(open(args.results, "rb"))
    K = np.array(json.load(open(args.camera))["cam_K"], dtype=float).reshape(3, 3)
    models_info = json.load(open(args.models_info))

    frames = []
    for key in sorted(results, key=lambda k: int(k.split("/")[1])):
        im_id = int(key.split("/")[1])
        rgb_path = os.path.join(args.rgb_dir, f"{im_id:06d}.png")
        img = cv2.imread(rgb_path)
        if img is None:
            print("missing rgb:", rgb_path); continue
        for det in results[key]:
            R = np.array(det["R"], dtype=float).reshape(3, 3)
            t = np.array(det["t"], dtype=float).reshape(3)
            ob_in_cam = np.eye(4); ob_in_cam[:3, :3] = R; ob_in_cam[:3, 3] = t
            bbox = bbox_from_models_info(models_info, det["obj_id"])
            img = draw_posed_3d_box(K, img, ob_in_cam, bbox)
            img = draw_xyz_axis(img, ob_in_cam, args.axis_scale, K)
        out_path = os.path.join(args.out, f"{im_id:06d}.png")
        cv2.imwrite(out_path, img)
        frames.append(img)
    print(f"wrote {len(frames)} overlays -> {args.out}")

    if args.video and frames:
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(os.path.join(args.out, "overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        print("wrote", os.path.join(args.out, "overlay.mp4"))


if __name__ == "__main__":
    main()
