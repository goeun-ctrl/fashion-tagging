# MAT4091 Fashion Category Tagging — submission

This file is filled in by `scripts/make_submission.py`. Items in [brackets] must be completed by the team before hand-in.

## Run

```
python predict.py --input_dir TEST_IMAGES --output predictions.jsonl
```

- Output: one JSON object per image, `{"test_id":"<file name without extension>","labels":["jacket","pants"]}`;
  `labels` holds exact category names and is `[]` when no category is predicted.
- Offline: nothing is downloaded. The weights and every setting are in `model.pt`; timm models are created with `pretrained=False`.
- Optional flags: `--device cpu|cuda`, `--batch_size N`, `--workers N` (0 = no loader processes), `--strict`, `--print_params`.
- An image that cannot be read gets `[]` and a warning on stderr (`--strict` exits with code 4 instead).

## Parameters

**Total parameters used to produce a prediction: @@PARAMS@@** (limit 50,000,000). One model, no ensemble.
`python predict.py --print_params` prints the count.

## Model

| Item | Value |
|---|---|
| Backbone | @@MODEL@@ (public ImageNet-22k pre-trained weights, fine-tuned) |
| Head | @@HEAD@@, gate output: @@GATE@@ |
| Input | EXIF transpose, RGB, keep aspect ratio, short side @@SHORT@@, long side <= @@MAXLONG@@, bilinear, padded to a fixed canvas; pooling over the real image area only |
| Test-time augmentation | @@TTA@@ |
| Thresholds | @@THRESHOLDS@@ (chosen on the internal holdout of the training data only) |
| Trained weights | run `@@RUN@@`, epoch @@EPOCH@@ (EMA weights) |
| Tested with | @@VERSIONS@@ |

## Training data and reproduction

[Fashionpedia train split provided on the LMS: 43,197 training images, 2,425 internal holdout images used for epoch and threshold
selection; the official validation images were used for reporting only. List any extra images with source, labels and use.]

[Exact training commands: scripts/train_hires.py ... — copy from the final run's config.json.]

Pre-trained weights used for initialisation: [source and licence of `convnext_tiny.fb_in22k_ft_in1k_384` from timm / Hugging Face].
The public test sample was not used for training, labelling, model selection or threshold selection.
