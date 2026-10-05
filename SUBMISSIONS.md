# Kaggle submissions — small track

One entry per Kaggle submission: what changed since the previous submission, the settings, and the results. Add an entry every time you submit. Scores are **public** leaderboard accuracy; the private leaderboard decides the final ranking.

- **Run folders:** `kaggle_runs/<timestamp>/runs/<RUN_NAME>/` (Kaggle runs, on this machine) or `MyDrive/eurosat/runs/<RUN_NAME>/` (Colab runs). Each has `config.json`, the weights and the CSV.
- **Class-mix gap:** how far the predicted test class mix is from EuroSAT's (Section 11 of the notebook). If the test set has EuroSAT's mix, accuracy is at most 1 − gap. Lower is better.

## Overview

| # | Time (UTC) | Run | Change vs. previous submission | Val acc | Mix gap | Public |
|---|---|---|---|---|---|---|
| 1 | 10-05 07:52 | `small_20261005_072538` | First real run (`NORM_MODE="train"`) | 0.9874 | — | 0.1418 |
| 2 | 10-05 10:42 | `small_20261005_103129` | `NORM_MODE="separate"` | 0.9878 | — | 0.4076 |
| 3 | 10-05 10:55 | `small_20261005_104424` | `AUG_RADIOMETRIC=True` (gain/offset 0.1/0.1) | 0.9856 | — | 0.6654 |
| 4 | 10-05 11:28 | `small_20261005_111722` | Repeat of the 12-band setup (details not recorded) | 0.9863 | — | 0.6593 |
| 5 | 10-05 11:45 | `small_20261005_113537` | Dropped B01, B09 (10 bands) | 0.9856 | — | 0.6069 |
| 6 | 10-05 12:25 | `small_20261005_121606` | 10 bands again (details not recorded) | 0.9865 | — | 0.5770 |
| 7 | 10-05 12:50 | `small_20261005_123950` | Back to 12 bands | 0.9848 | — | 0.6729 |
| 8 | 10-05 18:38 | `small_20261005_183041` | First run on Kaggle; same settings as #3 | 0.9867 | 0.239 | 0.6701 |
| 9 | 10-05 19:32 | `small_20261005_192446` | Stronger augmentation 0.2/0.3 (see below) | 0.9819 | 0.242 | **0.6954** |
| 10 | 10-05 19:49 | `small_20261005_192446` | Duplicate of #9 (two sessions submitted the same run) | 0.9819 | 0.242 | 0.6954 |
| 11 | 10-05 19:59 | `small_20261005_195110` | Augmentation 0.3/0.5 (stronger than #9) | 0.9800 | 0.212 | 0.6809 |
| 12 | 10-05 20:12 | `small_20261005_200237` | Back to 0.2/0.3, plus NDVI channel | 0.9824 | 0.278 | 0.6401 |
| 13 | 10-05 20:47 | `small_20261005_202928` | NDVI off; per-image scaling added (`PER_IMAGE="add"`) | 0.9843 | 0.209 | 0.5994 |

**What we have learned so far:**

- **Standardising test images separately** (#1 → #2) and **brightness augmentation** (#2 → #3) gave the big jumps.
- **Dropping B01/B09 hurt** (#5, #6 vs #3, #4, #7).
- **Stronger augmentation helped up to a point:** 0.2/0.3 gained +0.025 (#8 → #9), but 0.3/0.5 gave back 0.015 (#11).
- **The NDVI channel hurt** (#12, −0.055 vs #9).
- **Per-image scaling hurt badly** (#13, −0.096 vs #9), even though it makes the input immune to whole-image brightness changes.
- **The class-mix gap is only a rough guide:** #11 had the lowest gap so far but scored below #9. It did flag #12, the worst of the three, correctly.
- **Current notebook defaults = #9** (12 bands, `NORM_MODE="separate"`, augmentation 0.2/0.3, no NDVI).
- **Run-to-run noise:** identical settings varied by about ±0.007 (#3, #4, #7, #8), so differences smaller than ~0.015 are not meaningful.

## Details

### #1–#7 (Colab, before this log existed)

Reconstructed from the Kaggle submission messages. Only the band count, `NORM_MODE`, TTA and validation accuracy were recorded in those messages. The other settings are in `MyDrive/eurosat/experiment_log.csv` on Drive. #2 most likely ran without `AUG_RADIOMETRIC`, since #3 is the first run known to have it on, but this is not confirmed.

### #8 — `small_20261005_183041` (Kaggle run `kaggle_runs/20261005_203849`)

- **Settings:** 12 bands, `NORM_MODE="separate"`, `AUG_RADIOMETRIC=True` (gain 0.1, offset 0.1), TTA, 30 epochs, 588,042 parameters.
- **Result:** val 0.9867, public 0.6701. Predicted test mix: SeaLake 24 %, PermanentCrop 20 %, AnnualCrop 3 %. EuroSAT has 7–11 % per class.

### #9 / #10 — `small_20261005_192446` (Kaggle run `kaggle_runs/20261005_213252`)

Notebook changes since #8:

- **Augmentation strength is now a setting.** `AUG_GAIN` and `AUG_OFFSET` in Section 3 replace the hard-coded 0.1/0.1. This run used **gain 0.2, offset 0.3**.
- **Optional NDVI channel.** `ADD_NDVI` adds NDVI = (B08 − B04)/(B08 + B04) from the course's `cc_1` notebook as an extra input channel. It is saved with the normalisation statistics and handled by Section 13. **Off** in this run.
- **Class-mix gap.** Section 11 shows the predicted mix next to EuroSAT's and computes the gap; Section 12 logs it as `class_mix_gap`.
- **Experiment log.** It is now rewritten instead of appended to, so new settings columns can't misalign old rows.
- **Data loading.** `NUM_WORKERS = min(4, CPUs)` and persistent DataLoader workers. Epoch time on Kaggle stayed at 12–13 s, so loading wasn't the bottleneck.

Result: val 0.9819 (stronger augmentation makes training harder, −0.005), mix gap 0.242, **public 0.6954 (+0.025 vs #8)**. Section 13 reproduced the CSV exactly.

#10 is the same CSV submitted a second time: another Claude session pushed and submitted the shared kernel while this run was in progress. It cost one submission and carries no information.

### #11 — `small_20261005_195110` (Kaggle run `kaggle_runs/20261005_215912`)

- **Change since #9:** stronger radiometric augmentation, `AUG_GAIN` 0.2 → **0.3** and `AUG_OFFSET` 0.3 → **0.5**. Everything else unchanged (12 bands, `NORM_MODE="separate"`, `ADD_NDVI=False`, TTA, 30 epochs, 588,042 parameters).
- **Result:** val 0.9800, mix gap 0.212, **public 0.6809 (−0.015 vs #9)**. Section 13 reproduced the CSV exactly.
- **Conclusion:** keep 0.2/0.3. A likely explanation is that this much jitter starts to hide real spectral differences between classes; validation accuracy also dropped.

### #12 — `small_20261005_200237` (Kaggle run `kaggle_runs/20261005_221136`)

- **Change since #11:** augmentation back to #9's **0.2/0.3**, and **`ADD_NDVI=True`**: NDVI = (B08 − B04)/(B08 + B04), from the course's `cc_1` notebook, as a 13th input channel (588,330 parameters).
- **Result:** val 0.9824, mix gap 0.278, **public 0.6401 (−0.055 vs #9)**.
- **Conclusion:** NDVI stays off. A likely reason is that NDVI differs systematically between L1C and L2A (atmospheric correction raises it), so the extra channel adds shifted information rather than robust information.
- **Bug found:** Section 13 (reproduction) crashed in this run with `Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor)`. The NDVI statistics reloaded from `norm_stats.npz` are float64, and NumPy 2 then promotes the channel to float64. The CSV itself was written and format-checked before the crash. Fixed in the notebook by casting the channel to float32 and reading the statistics as plain floats.

### #13 — `small_20261005_202928` (Kaggle run `kaggle_runs/20261005_224734`)

- **Change since #12:** NDVI off again, and **`PER_IMAGE="add"`**. Besides the 12 standardised bands, the model gets each band scaled between that image's own 2nd and 98th percentile (`normalize_for_display` from the course's `cc_1` notebook): 24 input channels, 591,498 parameters. Radiometric jitter (0.2/0.3) only on the standardised channels.
- **Result:** val 0.9843, mix gap 0.209, **public 0.5994 (−0.096 vs #9)**. Section 13 reproduced the CSV exactly. Epochs took 32 s instead of 13 s (percentiles are computed per image on the CPU).
- **Submitted by:** another Claude session (submission message `small track | 20261005_224718`), which pushes and submits the shared kernel right after this notebook changes.
- **Conclusion:** off. A likely reason is that the per-image channels throw away absolute brightness, which separates classes such as SeaLake and Forest, and that the model leans on these channels because they are never jittered. `"replace"` (only per-image channels) was not tried, because it discards even more.

---

## Template for new entries

```markdown
### #N — `RUN_NAME` (Kaggle run `kaggle_runs/<timestamp>` or Colab)

Change since the previous submission: ...
Settings: bands, NORM_MODE, AUG_GAIN/AUG_OFFSET, ADD_NDVI, epochs, parameters
Result: val ..., mix gap ..., public ...
```

Also add a row to the overview table.
