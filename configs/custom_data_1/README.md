# custom_data_1 layout

The custom RGBD adapter reads either a manifest file or a simple folder layout.

Default layout:

```text
/home/pose/dipl/datasets/custom_data_1/
  rgb/
    000000.png
    000001.png
  depth/
    000000.png
    000001.png
  camera.json
```

Optional files:

- `annotations.json` or another manifest file with per-image records.
- `objects.txt` or `objects.json` if you want to declare object names for a known checkpoint.
- `models/` if you have mesh files for the object set.

`camera.json` can be either a single camera description or a per-image mapping. The simplest form is:

```json
{
  "cam_K": [
    600.0, 0.0, 320.0,
    0.0, 600.0, 240.0,
    0.0, 0.0, 1.0
  ],
  "depth_scale": 1000.0,
  "width": 640,
  "height": 480
}
```

If you only want to verify the pipeline, you can leave the dataset without annotations and run the detector/pose configs as-is. The pose stage will then emit empty results unless you later add detections through `DATASETS.DET_FILES_TEST`.