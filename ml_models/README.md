# ml_models/

Training output lands here. Weights are not committed - they are reproducible
from the datasets and the training scripts, and a checkpoint per run would bloat
the repository quickly.

## Producing a model

```bash
pip install -r requirements.txt -r training/requirements.txt
python training/download_datasets.py exit_signs_v2
python training/train_exit_detector.py
```

Layout after a run:

```
ml_models/exit_detector/
├── weights/
│   ├── best.pt          <- picked up automatically
│   └── last.pt
├── results.png
├── confusion_matrix.png
└── args.yaml
```

## Using a model

`PathFinder` takes the newest `ml_models/**/weights/best.pt` by default. Override
it with either:

```bash
export EPF_MODEL_PATH=/path/to/best.pt
python -m emergency_path_finder --model /path/to/best.pt --image photo.jpg
```

With no weights present the pipeline runs on classical computer vision alone and
says so on stderr.

## For the mobile app

The trainer exports `flutter_app/assets/models/exit_detector.tflite` when
TensorFlow is installed. That file is gitignored too - copy it in before
`flutter build`.
