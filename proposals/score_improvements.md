# Improving the Kaggle score: proposals

*Written 2026-10-05, based on run `small_20261005_072538` of [small_track_colab.ipynb](../small_track_colab.ipynb). None of the changes below have been run yet.*

The examples use the small-track notebook, but every proposal applies the same way to the medium track.

## TL;DR

- The model itself is fine: **98.7 % validation accuracy** (best checkpoint, with TTA).
- On the test set it fails: it predicted **3,918 of the 4,232 test images (93 %) as `HerbaceousVegetation`**. Unless the test set really is mostly that class, the Kaggle score is close to random guessing.
- The cause is the difference between the training images (EuroSAT, **Level-1C**, top of atmosphere) and the test images (**Level-2A**, atmosphere removed). Two bands (B01, B09) arrive at the model about **4 standard deviations** away from anything it saw in training.
- So the points are in **fixing the test preprocessing**, not in the architecture. The cheapest fixes need no retraining and can be tried today on the checkpoint we already have.

Recommended order: **check the band order → histogram matching + AdaBN (no retraining) → drop B01/B09 → atmospheric augmentation → self-training (after asking the TAs) → model tweaks.**

---

## 1. Why the current submission fails

From the Section 4 statistics, here is each test band compared with training. "Shift" is how far the test mean is from the training mean, measured in training standard deviations (σ). "Spread ratio" is test std ÷ train std.

| Band | Train mean | Test mean | Shift | Spread ratio |
|---|---:|---:|---:|---:|
| **B01** (aerosol, 60 m) | 1351 | 381 | **−3.9σ** | 1.31 |
| **B02** (blue) | 1116 | 404 | **−2.1σ** | 1.20 |
| B03 (green) | 1042 | 628 | −1.0σ | 1.21 |
| B04 (red) | 949 | 578 | −0.6σ | 0.99 |
| B05 | 1201 | 921 | −0.5σ | 1.14 |
| B06 | 1998 | 1805 | −0.2σ | 1.36 |
| B07 | 2365 | 2093 | −0.2σ | 1.29 |
| B08 | 2292 | 2180 | −0.1σ | 1.34 |
| B8A | 2589 | 2251 | −0.3σ | 1.22 |
| **B09** (water vapour, 60 m) | 731 | 2237 | **+3.7σ** | **3.63** |
| B11 | 1817 | 1455 | −0.4σ | 1.05 |
| B12 | 1118 | 937 | −0.2σ | 1.06 |

This matches the physics:

- **Visible bands get darker.** In L1C, light scattered by the atmosphere ("haze") adds a brightness offset, strongest in blue and weaker at longer wavelengths. Atmospheric correction removes it. That is the B01 → B05 pattern above. It is also visible in the Section 4 pictures: test water is almost black, while training water looks blue-grey.
- **B01 and B09 are atmospheric bands.** They exist to measure aerosols and water vapour, so correction changes them the most. B09 doesn't just shift; its spread is 3.6× larger, so shifting the mean alone can't fix it.
- **Near-infrared bands have about 1.3× the spread** in test. That is a multiplicative change, not an offset.

With `NORM_MODE = "train"` the network receives these values unchanged. Validation accuracy can't show any of this, because the validation images come from EuroSAT like the training images.

---

## 2. Before anything else: check the test band order

`TEST_BANDS` assumes the standard order `B01 … B08, B8A, B09, B11, B12`. B8A and B09 have nearly identical medians in the test data, so the band-profile plot **cannot tell whether they are swapped**. If they are, every proposal below works on the wrong band (dropping "B09" would drop B8A).

**Check:** B01 and B09 are recorded at 60 m and enlarged to 10 m, so they are blurrier than every other band. Add this to Section 4, before `del tr, te`:

```python
def edge_energy(a):
    # Neighbour-pixel differences relative to band spread: lowest for blurry 60 m bands (B01, B09).
    d = np.abs(np.diff(a, axis=1)).mean(axis=(0, 1, 2)) + np.abs(np.diff(a, axis=2)).mean(axis=(0, 1, 2))
    return d / a.std(axis=(0, 1, 2))

display(pd.DataFrame({"train": edge_energy(tr), "test": edge_energy(te)}, index=TEST_BANDS).round(3))
```

**Expected:** in both columns, B01 and B09 have the lowest values. If the **test** column is lowest at "B8A" instead of "B09", swap those two entries in `TEST_BANDS`. The *Data* tab on Kaggle may also state the order; if it does, it settles the question.

---

## 3. Proposals, best value first

### P1 — Fix the test preprocessing (no retraining)

These options reuse the existing checkpoint and change only how test images are prepared. Each costs one Kaggle submission and seconds of compute.

#### P1a — `NORM_MODE = "separate"` (already in the notebook)

This standardises test images with their own per-band mean and std, so the first two moments line up with training. It's the baseline for P1b.

#### P1b — Histogram matching

This is the non-linear version of P1a: it maps each test band onto the training distribution value by value, so the whole distribution lines up, not just the mean and std. That matters for B09, whose distribution shape is different. Put it in Section 5:

```python
def band_quantiles(images, idx, bands, n=500, q=np.linspace(0, 100, 1001)):
    # Per-band value distribution (1001 quantiles) over a random sample of images.
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))
    return np.percentile(images[pick][..., bands].astype(np.float32), q, axis=(0, 1, 2)).T  # (bands, 1001)

def hist_match(images, q_src, q_dst):
    # Map every band of `images` from the source distribution onto the destination one.
    out = np.empty(images.shape, np.float32)
    for b in range(images.shape[-1]):
        out[..., b] = np.interp(images[..., b], q_src[b], q_dst[b])
    return out

Q_TRAIN = band_quantiles(X, train_idx, KEEP_TRAIN)
Q_TEST  = band_quantiles(X_test, test_idx, KEEP_TEST)
X_test_hm = hist_match(X_test[..., KEEP_TEST], Q_TEST, Q_TRAIN)          # ~0.8 GB float32

# The bands are already selected and in USE_BANDS order, so take all of them; standardise with training stats.
test_ds = EuroSATDataset(X_test_hm, test_idx, list(range(len(USE_BANDS))), TRAIN_MEAN, TRAIN_STD)
test_loader = make_loader(test_ds, False)
```

Histogram matching also covers the `TEST_OFFSET` problem, so leave `TEST_OFFSET = 0` when using it.

**Caveat:** it assumes the test set has roughly the same mix of classes as EuroSAT. If the test set is, say, half forest, matching the whole distribution will distort it a little. It is still far better than the current 4σ gap.

#### P1c — AdaBN (re-estimate BatchNorm statistics on test images)

Every convolution in `SmallCNN` is followed by BatchNorm, whose running mean and variance were measured on EuroSAT. AdaBN replaces them with values measured on the test images. The weights stay unchanged and no parameters are added. For BatchNorm networks under a shift like this one it is often the largest single gain, and it combines with P1a or P1b.

Run it in Section 11, **after** Section 10 has computed `VAL_ACC`. After adaptation the model is tuned to the test images, so validation numbers measured afterwards are meaningless.

```python
def adapt_bn(model, ds):
    # AdaBN: replace the BatchNorm running statistics with ones measured on `ds`; weights unchanged.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats(); m.momentum = None   # None = exact average over all batches
    model.train()                                        # train mode makes BatchNorm collect statistics
    with torch.no_grad():
        for xb, _ in make_loader(ds, shuffle=True):     # shuffled, in case test ids are grouped by class
            model(xb.to(device))
    return model.eval()

model.load_state_dict(torch.load(CKPT, map_location=device))
adapt_bn(model, test_ds)
torch.save(model.state_dict(), RUN_DIR / "small_best_adabn.pt")   # these weights produce the CSV
sub = make_submission(model, test_loader, USE_TTA)
```

The loader **must be shuffled**: if the test ids happen to be ordered by class, unshuffled batches would each hold one class and the variance estimate would be too small.

### P2 — Drop B01 and B09 (needs retraining)

```python
USE_BANDS = [b for b in TEST_BANDS if b not in ("B01", "B09")]
```

These are the two worst bands in the table, and EuroSAT reaches about 98 % with RGB alone, so validation accuracy should barely move. **Do the band-order check (Section 2 of this document) first.** Combine with the best P1 option.

### P3 — An augmentation that imitates atmospheric correction (needs retraining)

The current `AUG_RADIOMETRIC` jitters values by about 0.1σ after standardisation, 20–40× smaller than the real shift. Instead, simulate the correction on the **raw** values, before standardisation:

1. **Haze estimate per band** (dark-object subtraction): the darkest pixels (water, shadow) should be near zero after correction. The gap between the 1st percentiles of training and test is an estimate of the haze. Reuse the quantiles from P1b: index 10 of the 1001 quantiles is the 1st percentile.

   ```python
   HAZE = np.clip(Q_TRAIN[:, 10] - Q_TEST[:, 10], 0, None).astype(np.float32)   # USE_BANDS order
   ```

2. **In `EuroSATDataset.__getitem__`**, for a share `AUG_ATMOS` of training images, subtract a random fraction of the haze and apply a random per-band gain (the test near-infrared spread is about 1.3×):

   ```python
   raw = self.images[i][..., self.bands].astype(np.float32) - self.offset
   if self.augment and self.haze is not None and torch.rand(1).item() < AUG_ATMOS:
       alpha = 1.2 * torch.rand(1).item()                              # 0 = L1C as is, ~1 = L2A-like
       gain = (0.8 + 0.5 * torch.rand(len(self.bands))).numpy()        # per band x0.8 .. x1.3
       raw = np.clip(raw - alpha * self.haze, 0, None) * gain
   img = (raw - self.mean) / self.std
   ```

   Add `haze=None` to the constructor and pass `haze=HAZE` only to `train_ds`. Use `torch.rand`, not `np.random`, as the existing augmentation does: it keeps the random numbers different in each DataLoader worker.

The model then sees both L1C-like and L2A-like versions of every training image. B09 gets a haze of 0 (its test values are *higher*), so this does not cover B09, which is another reason to combine it with P2.

### P4 — Self-training on the test images (ask the TAs first)

Once P1–P3 give a sensible spread of predicted classes:

1. Predict the test set with TTA and keep the images the model is very confident about (probability > 0.9). Take **at most K per class** (e.g. 300) so a dominant class can't take over.
2. Add them to training with their predicted labels, and fine-tune for a few epochs at a lower learning rate.
3. Repeat 2–3 rounds.

```python
probs = predict_proba(model, test_loader, tta=True)
conf, pred = probs.max(1), probs.argmax(1)
keep = np.concatenate([np.where(pred == c)[0][np.argsort(-conf[pred == c])][:300] for c in range(len(CLASSES))])
keep = keep[conf[keep] > 0.9]
pseudo = np.full(len(X_test), -1); pseudo[keep] = pred[keep]

# Same images, bands and statistics as test_ds (e.g. X_test_hm with TRAIN_MEAN/STD for P1b), now with labels and augmentation.
pl_ds = EuroSATDataset(X_test_hm, keep, list(range(len(USE_BANDS))), TRAIN_MEAN, TRAIN_STD,
                       labels=pseudo, augment=True)
train_loader = make_loader(torch.utils.data.ConcatDataset([train_ds, pl_ds]), True)   # fit() uses this global
```

This lets the model learn directly from test-domain images and usually adds a few points. Two risks:

- **Rules.** It uses the unlabelled test images during training. Confirm with the TAs that this is allowed before submitting a model trained this way.
- **Confirmation bias.** Wrong pseudo-labels get reinforced. The confidence threshold and the per-class cap limit this; check the predicted class spread after every round.

### P5 — Model-side changes (low priority)

Validation is already at 98.7 %, so there are at most about 1.3 points to gain in-domain. Worth doing once the shift is under control:

- **Label smoothing:** `nn.CrossEntropyLoss(label_smoothing=0.1)`. It gives less over-confident probabilities, which also helps the P4 threshold.
- **Mixed precision** (`torch.autocast` + `GradScaler`): roughly twice as fast on a T4, so more experiments per session.
- **Weight averaging** (EMA or SWA, `torch.optim.swa_utils`): costs nothing in parameters, since the final model is still one set of weights.
- **Width:** there is room from 588k to about 1M parameters. The assert in Section 7 stops the cell if a setting goes over the limit.
- **Avoid ensembles.** The parameter limit would most likely count every model in the ensemble together.

### Other ideas (not ranked)

- **Spectral indices as extra input channels**, e.g. NDVI = (B08−B04)/(B08+B04), NDWI = (B03−B08)/(B03+B08), NDBI = (B11−B08)/(B11+B08). Ratios cancel multiplicative changes but not the haze offset, so they help only together with P1–P3.
- **Per-image standardisation** (each image standardised with its own per-band mean and std). It removes most of the shift, but also the absolute brightness and the brightness ratios between bands, which separate e.g. water from forest. Worth one experiment, not a default.

---

## 4. Measuring progress without labels

Validation accuracy can't show the test-set problem, so use these signals together:

1. **How the predicted test labels spread over the classes.** Check this before every submission. EuroSAT has 7–11 % per class; something close to that is a good sign, and 93 % in one class is a broken model.

   ```python
   display(sub.label.value_counts(normalize=True).reindex(CLASSES, fill_value=0).round(3))
   ```

2. **Agreement between setups.** `(sub_a.label == sub_b.label).mean()`. Two good, different setups should agree on most images. Low agreement means at least one of them is still broken.

3. **The public Kaggle score**, used sparingly. The final ranking uses the **private** leaderboard, so tuning many small settings against the public score risks overfitting to it. Use it for the large decisions (P1 vs P2 vs P3), not for fine-tuning.

---

## 5. Reproducibility (rubric: "code reproduces csv file")

Every test-side change adds state that `reproduce_submission` (Section 13) must reload. For each technique we keep, save its state in `RUN_DIR` and extend Section 13 to use it:

| Technique | Save | Section 13 must |
|---|---|---|
| P1b histogram matching | `np.savez(RUN_DIR / "quantiles.npz", q_train=Q_TRAIN, q_test=Q_TEST)` | apply `hist_match` with the saved tables |
| P1c AdaBN | `small_best_adabn.pt` | load the adapted weights instead of `small_best.pt` |
| P3 augmentation | `HAZE` in `norm_stats.npz` | nothing at test time (record it anyway, for the slides) |
| All | new settings (`TEST_PREP`, `ADABN`, `AUG_ATMOS`) in `config.json` and the Kaggle message | read them from `config.json` |

The TAs will test the weights we hand in, so for AdaBN those must be the **adapted** weights.

---

## 6. Suggested experiment plan

Change one thing at a time and log each run in `experiment_log.csv` with its Kaggle score.

| Run | Change | Retrain? | Notes |
|---|---|---|---|
| E0 | Current baseline | — | Record the Kaggle score; it is the reference |
| E1 | `NORM_MODE = "separate"` | No | Existing option |
| E2 | Histogram matching (P1b) | No | |
| E3 | Best of E1/E2 + AdaBN (P1c) | No | Expected to be the largest jump |
| E4 | Drop B01, B09 (P2) + best of E1–E3 | Yes | After the band-order check |
| E5 | E4 + atmospheric augmentation (P3) | Yes | Try `AUG_ATMOS` 0.5 and 0.8 |
| E6 | E5 + self-training (P4) | Yes | Only after the TAs confirm it is allowed |
| E7 | E6 + label smoothing / EMA / width (P5) | Yes | |

E1–E3 cost three submissions and can be done today.

### Suggested new settings for Section 3

```python
TEST_PREP = "hist"   # CHOOSE: "train" | "separate" | "hist" (histogram matching; replaces NORM_MODE)
ADABN     = True     # CHOOSE: re-estimate BatchNorm statistics on the test images before predicting
AUG_ATMOS = 0.5      # CHOOSE: share of training images given simulated atmospheric correction (0 = off)
```

---

## 7. Open questions

- **Test band order:** confirm with the edge-energy check (Section 2 of this document) or Kaggle's *Data* tab.
- **Self-training:** ask the TAs whether training on unlabelled test images is allowed (P4).
