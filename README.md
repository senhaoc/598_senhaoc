````markdown
# VLM-Adapter for Low-Resource Medical Image Understanding

This repository contains a modular Python implementation of the experiments
described in the project **“VLM-Adapter: Low-Resource Medical Image
Understanding via Frozen Feature Reuse and Expert Prompting”**.

The original work was developed in a single Colab notebook; here it is
refactored into reusable Python modules organized as a standard GitHub
repository.

---

## Repository Layout

```text
vlm-adapter-med/
├── README.md
├── requirements.txt          # (optional) torch, torchvision, transformers, diffusers, sklearn, etc.
├── config.py                 # global config (DATA_ROOT, device, random seed)
├── data_utils.py             # data loading utilities and torch Dataset
├── clip_backbone.py          # CLIP loading helpers
├── clip_linear_experiments.py# zero-shot CLIP vs CLIP+linear probe + t-SNE
├── cnn_vs_clip_scaling.py    # k-shot scaling: CNN vs CLIP adapter
├── discussion_analysis.py    # robustness to noise, prompt sensitivity, heatmaps
├── generation_quality.py     # Stable Diffusion FID + CLIPScore, BLIP captions
├── system_pipeline.py        # end-to-end CLIP+BLIP+SD system evaluation
├── lesion_synthesis.py       # image-to-image diffusion lesion synthesis
├── baselines_cnn.py          # ResNet baselines vs CLIP+linear probe
└── advanced_experiments.py   # synthetic augmentation curve, prompt ablation, calibration
````

Each script can be run independently and corresponds to a major block of
experiments in the report.

---

## Requirements

Tested with:

* Python 3.10+
* PyTorch (with CUDA recommended)
* `torchvision`
* `transformers`
* `diffusers`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `pandas`
* `nltk`

You can install dependencies via:

```bash
pip install -r requirements.txt
```

(You should populate `requirements.txt` with the above packages and suitable
version pins for your environment.)

---

## Data

All experiments use the **Chest X-Ray Pneumonia** dataset from Kaggle:

> [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

1. Download the dataset from Kaggle.
2. Unzip it so the directory structure looks like:

```text
DATA_ROOT/
  train/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

3. Set the `DATA_ROOT` environment variable or edit `DATA_ROOT` in
   `config.py` to point to this path.

---

## How to Run

Below are the main entry points and what they reproduce:

### 1. Zero-shot CLIP vs CLIP + Linear Probe

```bash
python clip_linear_experiments.py
```

* Compares CLIP zero-shot classification under different prompts with a
  few-shot logistic regression head (CLIP+Linear).
* Outputs ROC curves, a confusion matrix, and a t-SNE visualization of
  CLIP image embeddings.

---

### 2. k-shot Scaling: CNN vs CLIP Adapter

```bash
python cnn_vs_clip_scaling.py
```

* Trains ResNet baselines and the CLIP+Linear adapter under
  ${20, 50, 100, 200, 500}$-shot settings.
* Plots the scaling law (AUC vs number of labeled samples), t-SNE and
  a confusion matrix for the CLIP model.

---

### 3. Robustness, Prompt Sensitivity, and Heatmaps

```bash
python discussion_analysis.py
```

* Evaluates robustness to Gaussian noise.
* Measures zero-shot prompt sensitivity across several text templates.
* Produces CLIP-based text–patch similarity heatmaps showing where the
  model “looks” for pneumonia.

---

### 4. Stable Diffusion Generation Quality and BLIP Captions

```bash
python generation_quality.py
```

* Compares **baseline** vs **expert** Stable Diffusion prompts for
  pneumonia X-rays.
* Computes FID and CLIPScore against real images.
* Demonstrates BLIP captions with and without medical prompting.

---

### 5. End-to-End System Evaluation

```bash
python system_pipeline.py
```

* Trains the CLIP+Linear classifier (few-shot).
* Uses BLIP to generate short radiology-style impressions.
* Reports AUC, sensitivity/specificity, BLEU score for captions, and
  inference latency.
* Shows one qualitative example with image, prediction and text.

---

### 6. Lesion Synthesis (Image-to-Image Diffusion)

```bash
python lesion_synthesis.py
```

* Loads a real normal X-ray.
* Uses Stable Diffusion in image-to-image mode to add a unilateral
  pneumonia-like opacity.
* Produces a 4-panel figure (input, output, diff-heatmap, zoom-in)
  suitable for inclusion in the paper.

---

### 7. CNN Baselines and ROC Curves

```bash
python baselines_cnn.py
```

* Trains ResNet-18 baselines (from scratch and ImageNet-pretrained) on
  500-shot subsets.
* Trains CLIP+Linear on the same data.
* Prints a markdown table comparing AUC, sensitivity and specificity.
* Plots ROC curves of CNN baselines vs CLIP+Linear.

---

### 8. Advanced Experiments

```bash
python advanced_experiments.py
```

* Runs additional analyses such as the synthetic augmentation curve
  (adding 0/50/100/200 synthetic samples) and prompt ablation on the
  generation side.
* Produces figures similar to the ablation plots in the report.

---

## Notes

* All CLIP, BLIP and Stable Diffusion models are used in **frozen** mode;
  only small adapters (e.g., the logistic regression head) are trained.
* Some scripts, especially those using diffusion models, require a GPU
  with sufficient VRAM (e.g., 16 GB or more) for comfortable runtime.
* The code is organized to be easy to extend: you can plug in alternative
  backbones, add new prompts, or adapt the pipeline to other medical
  datasets by modifying `config.py` and `data_utils.py`.

If you use this code or ideas in academic work, please consider citing the
corresponding report on **VLM-Adapter for Low-Resource Medical Image
Understanding**.
