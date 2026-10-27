# Locating Photos by Country

MAAI901 Deep Learning for Visual Data, Assessment 2. Ahmad Hello (7573297).

I fine-tuned CLIP ViT-B/16, an ImageNet-21k ViT-B/16 and ResNet-50 to predict which of 25 countries a street-view photo was taken in, using the OSV-5M dataset. The full method and discussion are in my report.

## Files

- `MAAI901_W6_geolocation.ipynb`: the Colab notebook I ran, with its outputs. `MAAI901_W6_geolocation.pdf` is the same notebook as a PDF.
- `Visualisations_and_sample_outputs.pdf`: the result tables, figures and sample predictions.
- `src/`: the Python code used by the notebook (data preparation, training and analysis).
- `data/splits.csv.gz`: the image IDs of my training, validation and test splits.
- `results/`: the test results, the training log and my by-eye check of the scene labels.
- `figures/`: the figures. `fig7_samples_report.jpg` is `fig7` with larger labels, and `fig4` was saved again with a wider margin.
- `hf_model_checks/`: the official Hugging Face example notebooks of the three models, which I ran to check them.

## Running it

Open the notebook in Google Colab with a T4 GPU and run all cells. It saves its work to Google Drive. The images are not in this repo; the notebook downloads the ones listed in `data/splits.csv.gz` from OSV-5M.

## Results

Test accuracy (%), mean of three seeds:

| Model | All | Urban | Natural | Unseen contributors |
|---|---|---|---|---|
| CLIP ViT-B/16 | 75.5 | 74.9 | 72.1 | 53.8 |
| ViT-B/16 (ImageNet-21k) | 64.3 | 58.9 | 60.8 | 37.5 |
| ResNet-50 | 62.1 | 55.7 | 58.6 | 35.9 |

Urban and natural are averaged over the 21 countries with enough photos of both.

CLIP accuracy by country (%): New Zealand 95, Botswana 95, Zambia 94, Finland 88, Australia 88, Thailand 86, United Kingdom 85, Japan 84, Norway 81, Canada 81, Türkiye 78, Spain 77, India 77, Sweden 75, France 74, Brazil 73, Mexico 72, Russia 68, South Africa 68, United States 67, Tanzania 64, Peru 63, Argentina 55, Chile 52, Indonesia 50. The other two models' results by country are in `results/results.json`.

## Model card

- Use: research on how pre-training affects recognising countries from street-view photos. It should not be used on private photos, to find where people are, or to act on a prediction without a person checking it. The fine-tuned weights are not shared.
- Limits: about 85% of test photos come from contributors who are also in the training sample, so the accuracies are optimistic (see the unseen contributors column). The urban labels are noisy, and Botswana, Finland, South Africa and Zambia are left out of the urban and natural averages, as they have too few photos of one type. ResNet-50 may be under-tuned, as its best learning rate was the highest one I tried.
- Fairness: accuracy is uneven between countries, and many mistakes go to a neighbouring country, so a wrong answer can look believable.

## Datasheet

- Source: OSV-5M on Hugging Face, street-view photos from Mapillary under CC BY-SA.
- Sample: 25 countries with 1,500 training, 200 validation and 1,000 test photos each. At most 10 photos per sequence (5 in test), except Botswana's training photos, as it has too few sequences.
- Labels: the country comes from OSV-5M. Urban or natural comes from the Places365 scene classifier. When I checked 100 of each by eye, 97 natural labels were right but only 47 urban ones.
- Gaps: an equal sample per country hides how unevenly Mapillary covers the world. Faces and number plates are blurred, but not always, and the people in the photos did not agree to be filmed.

## Sources

- OSV-5M: https://huggingface.co/datasets/osv5m/osv5m
- CLIP ViT-B/16: https://huggingface.co/openai/clip-vit-base-patch16
- ViT-B/16 (ImageNet-21k): https://huggingface.co/google/vit-base-patch16-224-in21k
- ResNet-50: https://huggingface.co/microsoft/resnet-50
- Places365 scene classifier: http://places2.csail.mit.edu/models_places365/
