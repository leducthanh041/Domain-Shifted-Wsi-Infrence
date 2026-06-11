# MergeSlide-TTA

MergeSlide-TTA is a test-time adaptation extension for MergeSlide on whole-slide image inference.  
The method follows the MergeSlide-TTA research idea and adapts the merged model at test time by updating selected LayerNorm affine parameters, while keeping the merging structure and prompt embeddings frozen.

## 1. Requirements

Install the runtime stack used by this repo:

```text
numpy
pandas
scipy
scikit-learn
h5py
tqdm
omegaconf
torch
torchvision
transformers
matplotlib
seaborn
tensorboard
```

See `requirements.txt` for the full list.

## 2. Project Structure

```text
MergeSlide_TTA_v1/
├── README.md
├── docs/
├── mergeslide_tta/
│   ├── __init__.py
│   ├── constants.py
│   ├── datasets.py
│   ├── metrics.py
│   ├── model.py
│   ├── prompts_zeroshot.py
│   ├── tta_adapter.py
│   ├── tta_losses.py
│   └── utils.py
├── scripts/
│   ├── test_classIL.sh
│   ├── test_classIL_tta.sh
│   ├── test_taskIL.sh
│   └── test_taskIL_tta.sh
├── checkpoints -> /docker/data/thanhld/MergeSlide_TTA_v1/checkpoints
├── checkpoints_ood -> /docker/data/thanhld/MergeSlide_TTA_v1/checkpoints_ood
├── logs -> /docker/data/thanhld/MergeSlide_TTA_v1/logs
├── merge.py
├── opcm_mergeslide.py
├── task_prompts.pt
├── test_classIL_task_prompt.py
├── test_classIL_task_prompt_other_metrics.py
├── test_classIL_tta.py
├── test_taskIL.py
├── test_taskIL_tta.py
├── train.py
└── tools/
    └── run_classil_with_pt_features.py
```

## 3. Datasets

The experiments use six TCGA tasks:

- TCGA-BRCA
- TCGA-RCC
- TCGA-NSCLC
- TCGA-ESCA
- TCGA-TGCT
- TCGA-CESC

### Data preparation

Prepare WSI annotations and pre-extracted features, then point `mergeslide_tta/datasets.py` to the local dataset root. BRCA, RCC, and NSCLC use CSV-based split metadata; ESCA, TGCT, and CESC use the simplified directory-based format already supported by the repo.

## 4. Implementation

### 4.1. Base MergeSlide workflow

Per-task finetuning:

```bash
bash scripts/finetune.sh
```

Model merging:

```bash
bash scripts/mergemodel.sh
```

### 4.2. TTA evaluation

Class-IL TTA:

```bash
bash scripts/test_classIL_tta.sh
```

Task-IL TTA:

```bash
bash scripts/test_taskIL_tta.sh
```

These scripts run the current TTA setup used in this repo, including LN adaptation and the routing-aware variants introduced for MergeSlide-TTA.

### 4.3. Baseline evaluation entrypoints

Class-IL baseline on the original MergeSlide setting:

```bash
bash scripts/test_classIL.sh
```

Task-IL baseline on the original MergeSlide setting:

```bash
bash scripts/test_taskIL.sh
```

## 5. MergeSlide-TTA summary

The proposed adaptation strategy is:

- freeze merging coefficients
- freeze class-aware and task-level prompt embeddings
- freeze the TITAN backbone
- update only selected LayerNorm affine parameters during test time
- use entropy-guided confidence filtering to decide whether to adapt a slide
- keep the adaptation small enough to avoid breaking the merged model structure

This is intended for OOD / cross-site WSI inference, where the original MergeSlide model can lose accuracy under domain shift.

## 6. Acknowledgement

This project builds on the original MergeSlide code base:

- https://github.com/caodoanh2001/MergeSlide

It is also inspired by:

- TITAN
- FusionBench
- CATE

The authors thank the original projects for their work and for making the research path possible.
