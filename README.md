# Locating Photos by Country (MAAI901, Assessment 2)

Code for the Week 6 Applied Vision Transformer Report: CLIP ViT-B/16 vs. ImageNet-21k ViT-B/16
(ResNet-50 as a reference), fine-tuned to name the country of a street-view photo (25 countries, OSV-5M).

## What is in this folder

| Path | What it is |
|---|---|
| `MAAI901_W6_geolocation.ipynb` | The notebook that was run in Google Colab, with its outputs. Sections A to G: splits, image cache, scene labels, learning-rate search, final training, test predictions, results and figures. |
| `MAAI901_W6_geolocation.pdf` | The same notebook printed to PDF, with its outputs. |
| `Visualisations_and_sample_outputs.pdf` | The visualisations and sample outputs in one file: all result tables and figures, the sample predictions on test photos, and a summary of the scene-label check. |
| `src/data_prep.py` | Builds the balanced splits and the 256 x 256 image cache from the OSV-5M archives on Hugging Face. |
| `src/train_eval.py` | Models (loaded from the official Hugging Face checkpoints), data loading, two-stage fine-tuning, Places365 scene classifier. |
| `src/analysis.py` | Metrics, McNemar's test (H1), sequence bootstrap (H2), confusions. |
| `data/splits.csv.gz` | The image IDs of the training, validation and test splits (checksum `3ec8dc9f836b`). |
| `results/results.json` | All test results used in the report (3 seeds per model). |
| `results/log.txt` | The full run log (times in UTC): splits, cache, scene labels, learning-rate search and every training epoch. |
| `results/lr_choice.json` | The learning rate chosen for each model. |
| `results/model_check.json` | Weight check against the official Hugging Face files, and the file versions used. |
| `results/scene_check_by_eye.csv` | My verdicts for the 100 urban and 100 natural scene labels checked by eye. |
| `figures/` | All figures made by the notebook (in the report, `fig5` is Figure 1, `fig1` to `fig4` are Figures 2 to 5 and `fig7` is Figure 6, as `fig7_samples_report.jpg`; `fig6` and `fig8` are extra), the two scene-check contact sheets, and `scene_check_summary.png`, which sums up that check with four examples of each case. `fig7_samples_report.jpg` is `fig7` redrawn with larger labels for the report (Figure 6), and `fig4` was saved again with a wider left margin so its axis label is not cut off. |
| `hf_model_checks/` | The official "Open in Colab" notebooks of the three models, run without changes (`.ipynb` with outputs, and PDF). All local-inference cells ran without errors. The ResNet-50 notebook also has two remote-inference cells that need a personal Hugging Face token, so I did not run them. |

The three `src` files are the same code that the notebook writes with `%%writefile`.

## How to run it

1. Open `MAAI901_W6_geolocation.ipynb` in Google Colab and choose a T4 GPU.
2. Run all cells. The notebook mounts Google Drive and keeps everything in `MyDrive/MAAI901_W6`.

The images are not included here (OSV-5M on Hugging Face, CC BY-SA). The notebook rebuilds the splits from the
OSV-5M metadata, checks that they match `data/splits.csv.gz` (checksum `3ec8dc9f836b`), and downloads only those images.

In section C, the long `GROUP_CODES` string gives the Places365 scene hierarchy, one digit per scene category:
4 = indoor, 2 = outdoor natural, 1 = outdoor man-made, and sums such as 3 for categories in two groups.

## Model card (three fine-tuned classifiers, weights not released)

- **Models:** CLIP ViT-B/16 image encoder, ViT-B/16 pre-trained on ImageNet-21k, and ResNet-50, each with one linear
  head for 25 countries. Two-stage fine-tuning, learning rate chosen per model, seeds 0, 1 and 2.
- **Intended use:** research on how pre-training affects country recognition from street-view photos. As in CLIP's own
  model card, any deployed use is out of scope, and so are locating people, private photos, and using a prediction
  without a person checking it.
- **Data:** the balanced OSV-5M sample described in the datasheet below.
- **Results (mean of three seeds, %):**

  | Model | Top-1 | Macro-F1 | Unseen contributors | Urban | Natural |
  |---|---|---|---|---|---|
  | CLIP ViT-B/16 | 75.5 | 75.2 | 53.8 | 74.9 | 72.1 |
  | ViT-B/16 (ImageNet-21k) | 64.3 | 63.8 | 37.5 | 58.9 | 60.8 |
  | ResNet-50 | 62.1 | 61.6 | 35.9 | 55.7 | 58.6 |

  Urban and natural are averaged over the 21 countries with enough photos of both types.
- **CLIP recall by country (%, ISO codes as in Table 1 of the report):** NZ 95.0, BW 94.5, ZM 94.0, FI 88.4, AU 88.0, TH 86.3, GB 84.6, JP 84.0, NO 81.0, CA 80.6, TR 78.2, ES 77.0, IN 76.7, SE 74.5, FR 73.6, BR 72.7, MX 71.9, RU 68.5, ZA 67.9, US 67.2, TZ 63.6, PE 63.1, AR 54.8, CL 51.7, ID 50.2. The other two models are in
  `results/results.json` and `figures/fig3_country_recall.png`.
- **Limitations:** about 85% of test photos come from contributors who are also in the training sample, so the
  accuracies are optimistic (see the unseen-contributor column). Validation accuracy was about 10 points above test
  accuracy. The urban scene labels are noisy, and Botswana, Finland, South Africa and Zambia have no scene results.
  ResNet-50's best learning rate was the highest one tried.
- **Ethical notes:** geolocation is dual-use. Errors are uneven between countries and often fall on neighbours, so a
  wrong answer can look plausible.

## Datasheet (the OSV-5M sample used here)

- **Source:** OSV-5M (`osv5m/osv5m` on Hugging Face), street-view photos from Mapillary contributors, CC BY-SA.
  The images are not redistributed; `data/splits.csv.gz` lists the image IDs.
- **Composition:** 25 countries, 1,500 training, 200 validation and 1,000 test photos per country (67,500 in total).
  Training and validation come from the OSV-5M training split (different sequences), and test from the official
  test split.
- **Sampling:** at most 10 photos per sequence for training and validation and 5 for test. Botswana has too few
  sequences, so its training sample has no cap. Validation sequences were removed from training. The split was drawn
  with a fixed seed and is checked with a checksum.
- **Labels:** the country comes from OSV-5M. Urban and natural labels come from the Places365 ResNet-50 scene
  classifier and are noisy: in my check by eye, 97 of 100 natural labels were right, but only 47 of 100 urban labels
  were clearly right.
- **Known gaps:** the balanced sample hides how unevenly Mapillary covers the world. Mapillary blurs faces and number
  plates automatically, but not perfectly, and the people in the photos did not agree to be filmed.

## Sources

- OSV-5M dataset: https://huggingface.co/datasets/osv5m/osv5m
- Models: https://huggingface.co/openai/clip-vit-base-patch16, https://huggingface.co/google/vit-base-patch16-224-in21k, https://huggingface.co/microsoft/resnet-50
- Places365 ResNet-50 scene classifier: http://places2.csail.mit.edu/models_places365/
