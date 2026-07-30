# datasets/

Downloaded datasets land here. Nothing in this directory is committed - the four
datasets total roughly 2 GB.

## Getting them

```bash
python training/download_datasets.py --list   # what is registered, what is present
python training/download_datasets.py          # needs ROBOFLOW_API_KEY
```

Without an API key the script prints per-dataset manual instructions. Export in
**YOLOv8** format and extract so that `datasets/<key>/data.yaml` exists.

| Key | Dataset | Images |
|---|---|---|
| `exit_signs_v2` | [Emergency Exit Signs v2](https://universe.roboflow.com/emergency-exit-signs/emergency-exit-signs-v2) | ~1,070 |
| `stairs_detection` | [Stairs Detection](https://universe.roboflow.com/stairs-detection/stairs-fo4v5) | ~7,890 |
| `escalator_stairs` | [Escalator-Stairs](https://universe.roboflow.com/escalatorstairsdetection/escalator-stairs) | ~8,690 |
| `exit_detection` | [Exit-Detection](https://universe.roboflow.com/project1exits/exit-detection-w00yi) | ~36 |

## Expected layout

```
datasets/exit_signs_v2/
├── data.yaml
├── train/images/ train/labels/
├── valid/images/ valid/labels/
└── test/images/  test/labels/
```

`images/train` instead of `train/images` also works - `write_data_yaml()` detects
both.

## Storing them elsewhere

```bash
export EPF_DATASETS_DIR=/mnt/big-disk/epf-datasets
```
