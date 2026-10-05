# CLAUDE.md

## Project

EuroSAT land-cover coding challenge for the HSG course 7,854,1.00 Machine Learning (Fall 2026). Team of three, scored on Kaggle by accuracy; the **private** leaderboard decides the ranking.

- **Training data:** EuroSAT MS: Sentinel-2 **L1C** (top of atmosphere), 13 bands, 64×64, 10 classes.
- **Test data:** 4,232 Kaggle images: **L2A** (atmosphere removed), 12 bands (no B10), `.npy`.
- **Tracks:** small ≤ 1M parameters, medium ≤ 10M. Both tracks need a Kaggle submission, the code and the model weights.
- **Dates:** mid-term check-in 9 Nov 2026; final Kaggle submission, code and slides due 13 Dec 2026, 23:59; presentation 14 Dec.
- **Brief:** `ML-2026-Lab-02-Coding-Challenge.pdf` (grading rubric on slide 21).

Files:

- `small_track_colab.ipynb`: the small-track pipeline. It runs in Colab, with data cached on Google Drive (`MyDrive/eurosat/`).
- `proposals/`: written improvement proposals.
- `ML2026-Lab/`: git submodule with the official course labs. Read-only; never edit it.

## Rule: use the course's tools and methods

The notebooks must use the tools and methods taught in the course repository `ML2026-Lab/` (https://github.com/HSG-AIML-Teaching/ML2026-Lab).

- **Before adding or changing a method,** update the submodule (see below) and look up how the labs do it. Follow their API choices and style.
- **If the course doesn't cover a step yet,** keep the code standard and minimal, and note that it must be aligned once the matching lab is published.
- **Never quietly add a technique the course doesn't teach.** Name it, say the course doesn't cover it, and let the user decide.
- **When a new lab lands,** compare the notebook with it, propose alignments, and update the table below.

What the course covers so far (as of 2026-10-05: labs 101–103, `cc_1`, Lab 2):

| Step | Course tool | Where |
|---|---|---|
| Read EuroSAT GeoTIFFs | `rasterio`: `rio.open(...).read()` + `rasterio.plot.reshape_as_image` | `cc_1` |
| Display multispectral images | `normalize_for_display`: each band scaled between its 2nd and 98th percentile | `cc_1` |
| Band arithmetic | NDVI = (B8 − B4) / (B8 + B4) | `cc_1` |
| Train/evaluation split | `sklearn.model_selection.train_test_split`, `random_seed = 42` | Lab 2 |
| Feature scaling | `sklearn.preprocessing.MinMaxScaler` | Lab 2 |
| Classical classifier | `sklearn.svm.SVC` (linear / rbf / poly, `C`, `gamma`); HOG features with `skimage.feature.hog` | Lab 2 |
| Evaluation | `metrics.accuracy_score`, `classification_report`, `confusion_matrix` drawn with `sns.heatmap(mat.T, ..., cmap='BuGn_r')` | Lab 2 |
| Libraries | numpy, pandas, matplotlib, seaborn, scipy, scikit-image, PIL, torch, torchvision, rasterio (`ML2026-Lab/requirements.txt`) | — |

Planned, per the course README: Lab 3 PyTorch intro and MLP (5 Oct); Lab 4 custom datasets and CNNs (12 Oct); Lab 5 RNN/LSTM; Lab 6 attention; Lab 7 k-means/EM; Lab 8 autoencoders; Lab 9 transfer learning and self-supervised learning (7 Dec).

**Written before the PyTorch labs.** Notebook Sections 6–8 (Dataset/DataLoader, CNN, training loop) predate Labs 3–4. Align them once those labs are published.

**Not seen in the course yet; flag before relying on them:**

- **In the notebook already:** the AdamW optimiser, cosine learning-rate schedule, BatchNorm and test-time augmentation (TTA).
- **Proposed in `proposals/score_improvements.md`:** histogram matching, AdaBN, pseudo-labelling, EMA and mixed precision.

**Known discrepancy:** `cc_1` lists the EuroSAT band order as B1–B8, B8A, B9–B12, but the data stores **B8A last**. `TRAIN_BANDS` in Section 3 was checked against the data; keep it.

## Updating the course material (weekly)

```bash
make status    # checked-out course commit, and how many new commits are available
make update    # pull the latest labs (tracks main), list what changed, stage the submodule
git commit -m "Update course material"
```

After a fresh clone, run `make setup` (or clone with `--recurse-submodules`).

## Notebook conventions

- **Section headings:** sections are markdown headings `## N. Title`. `run_sections(...)` finds cells by these numbers, so keep the format.
- **Settings:** all settings live in Section 3 and are saved to each run's `config.json`. **CHECK** marks output to verify; **CHOOSE** marks a setting to experiment with.
- **Parameter limit:** Section 7 asserts the limit (≤ 1,000,000 for the small track). Keep it; the TAs check it.
- **Reproducibility:** Section 13 must rebuild the submitted CSV from the run's saved files (rubric). Anything a new test-time step needs (statistics, lookup tables, adapted weights) must be saved in `RUN_DIR` and loaded there.
- **Saving `.npz` files:** write them to Drive only with `save_npz` (Section 0). A plain `np.savez` on the Drive mount corrupted `norm_stats.npz` once.
- **Band selection:** `USE_BANDS` may only contain bands listed in `TEST_BANDS`. Drop bands there, not in `KEEP_TRAIN` / `KEEP_TEST`.
- **Judging changes:** judge changes aimed at the train/test difference by how the predicted test labels spread over the classes and by the Kaggle score. Validation accuracy can't see the L1C → L2A shift.

## Environment

The notebook runs on Colab with a T4 GPU, and the data lives on the user's Google Drive. This machine has neither the data nor torch or pandas, so notebook changes can't be run here. Check that edited cells parse, test pure-numpy helpers locally, and say clearly what is untested.

The notebook is stored as Colab JSON (indent 2, no trailing newline). Edit it through a JSON load and dump, not by hand.
