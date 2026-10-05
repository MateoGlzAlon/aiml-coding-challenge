# Running the small track on Kaggle

Develop `small_track_colab.ipynb` locally; Claude (or you) pushes it to a Kaggle
kernel with a GPU, waits for it to finish, and pulls back the results
(config, history, checkpoint, submission CSV, experiment log). No Colab, no
manual uploads. This exists because Kaggle has a scriptable API and free GPU
quota, whereas Colab only has a browser UI.

## One-time setup

1. **Kaggle API token.** Go to [kaggle.com/settings](https://www.kaggle.com/settings)
   → API → "Generate New Token", then save it yourself (not through Claude,
   so it never lands in a chat transcript):

   ```bash
   mkdir -p ~/.kaggle && cat > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
   ```

   Paste the token, press `Ctrl+D`.

2. **Phone verification.** Kaggle requires it before a kernel can use the
   internet or a GPU, regardless of what `kernel-metadata.json` asks for -
   without it every kernel fails instantly with
   `socket.gaierror: [Errno -3] Temporary failure in name resolution`, even
   though the metadata says `enable_internet: true`. Verify at
   [kaggle.com/settings](https://www.kaggle.com/settings) → "Phone verify",
   then confirm Settings → Internet is "Internet connected".

3. **Accept the competition rules** (needed to download the test set below):
   open the competition page on Kaggle and accept, if you haven't already.

4. **Build and publish the training + test data cache** (one-time,
   ~10 minutes):

   ```bash
   make kaggle-cache
   ```

   This downloads `EuroSAT_MS.zip` from Zenodo and decodes all 27,000
   GeoTIFFs **on a Kaggle kernel**, not locally - the decoded array is ~2.7 GB
   in RAM, more than this machine reliably has free. The result is published
   as a private Kaggle Dataset (`mateogonzalezalonso/eurosat-ms-l1c-cache`)
   that the training kernel attaches read-only, so this expensive step only
   happens once.

   The competition's own auto-mount (`competition_sources` in
   `kernel-metadata.json`) turned out not to actually attach any files in
   testing (see Troubleshooting), so the Kaggle test set is cached into the
   *same* dataset too - `make kaggle-cache` only builds the training half;
   run this once, locally, to add the test half (needs `.venv` and your
   token from step 1, but nothing from step 4's kernel):

   ```bash
   .venv/bin/kaggle competitions download -c 7-854-1-00-machine-learning-2026-coding-challenge -p /tmp/eurosat_test
   unzip -q /tmp/eurosat_test/*.zip -d /tmp/eurosat_test/raw
   # build X_test.npy/test_ids.npy from /tmp/eurosat_test/raw the same way cell 8 does,
   # then copy them + sample_submission.csv into kaggle/cache_dataset/ and:
   .venv/bin/kaggle datasets version -p kaggle/cache_dataset -m "add test cache" -d
   ```

   (This is a one-off; a helper script isn't wired up for it yet since it's
   only needed once per team. Ask Claude to do it if you need to rebuild.)

   Run `make kaggle-status` any time to check what's done.

## The iterate loop

```bash
.venv/bin/python scripts/kaggle_run.py
```

This copies the current `small_track_colab.ipynb` into `kaggle/small_track/`,
pushes it as a new kernel version (GPU on, the cache dataset attached
read-only), polls until it finishes, and downloads everything it wrote into
`kaggle_runs/<timestamp>/` - `config.json`, `history.csv`, the checkpoint,
`submission_small.csv`, plots, and the kernel's log (read that one for a
traceback if a run errors out).

Edit Section 3 (settings) between runs the same way you would for Colab -
`SMOKE_TEST`, bands, model width, epochs, etc. The notebook detects Kaggle via
`os.path.isdir("/kaggle/working")` and adapts automatically: reads the cached
training and test data from the attached dataset (found by filename search,
not a fixed path - see Troubleshooting), and writes everything to
`/kaggle/working`.

Training **never submits to Kaggle on its own** - `SUBMIT` is forced `False`
inside the kernel. Submit deliberately, after looking at a run's output:

```bash
.venv/bin/python scripts/kaggle_run.py --submit
# or: make submit
```

Mind the daily submission cap.

Useful flags: `--timeout SECONDS` (give up waiting, default 3600),
`--poll SECONDS` (status-check interval, default 30).

## Checking status

```bash
make kaggle-status
```

Reports auth, competition access, the cache kernel/dataset, and whether the
training kernel has run before. Kaggle's API only exposes a coarse state
(QUEUED/RUNNING/COMPLETE/ERROR/CANCELLED) for a kernel, never a percentage -
for a RUNNING kernel this prints elapsed time instead, which is the closest
real signal available from the CLI. For the actual progress lines (e.g.
`decoded 15000/27000`), watch the kernel's own page on kaggle.com; its logs
stream live there and nowhere else.

## Troubleshooting

- **A kernel fails instantly with a DNS error** - phone verification (step 2
  above) isn't done yet, or Internet isn't toggled on in Settings.
- **A run sits at "Running for Ns" with no new log lines** - could be a
  genuinely slow transfer, or a truly stalled connection with nothing printed
  about it. `kaggle/build_cache/build_eurosat_cache.py` sets a 60s socket
  timeout specifically so a stall raises and retries instead of hanging
  forever; if you add other network calls to a kernel script, give them the
  same protection.
- **Kernel status stays RUNNING well after the script's own log looks
  finished** - this is normal for a kernel with a large output: after the
  script exits, Kaggle still has to commit the contents of `/kaggle/working`
  to storage, which is a separate, slower, and invisible phase (no logs, no
  API field for it). Keep anything not meant to be output - temp downloads,
  extracted files - **outside** `/kaggle/working` (e.g. `/tmp`), or this
  phase gets dragged out for no reason, and `kaggle kernels output` can
  break entirely (next point).
- **`kaggle kernels output` downloads nothing, silently, even though the run
  completed** - happened once a kernel's output grew past ~500 files
  (27,000 extracted GeoTIFFs had ended up inside `/kaggle/working` by
  mistake). This CLI version's pagination is broken past the first page, and
  `--file-pattern` doesn't filter either - the only real fix is keeping a
  kernel's output to a handful of files.
- **A dataset you attach via `dataset_sources` doesn't mount where you
  expect** - we observed it at both `/kaggle/input/<slug>` and
  `/kaggle/input/datasets/<slug>` across otherwise-identical kernels, with no
  obvious trigger for which one you get. The notebook's `find_kaggle_input()`
  (cell 4) searches for the file by name under `/kaggle/input` rather than
  assuming a path - reuse it for anything else you attach.
- **`competition_sources` in `kernel-metadata.json` doesn't mount anything**
  - confirmed with a minimal reproduction: correct slug, correct format (no
  `c/` prefix, matches Kaggle's own docs), `/kaggle/input` simply doesn't
  contain the competition's files, in a script kernel, a notebook kernel, and
  with GPU on or off. The workaround here is caching the competition's test
  set into the same dataset as the training cache (step 4) instead of relying
  on the auto-mount at all.
- **`kaggle datasets status`/`list -m`/`files` says a just-published version
  isn't there yet** - eventual consistency; it shows up within ~15-30s.
- File sizes reported by `kaggle kernels files` / `kaggle datasets files`
  are sometimes wrong (we saw a 2.7 GB file reported as 859 bytes); don't
  trust that column, trust what actually lands on disk after downloading.

## Layout

| Path | What |
|---|---|
| `small_track_colab.ipynb` | the pipeline, single source of truth (same file Colab would use) |
| `kaggle/build_cache/` | one-off script kernel that builds the training-data half of the cache |
| `kaggle/cache_dataset/` | metadata for the published cache dataset (large files gitignored) |
| `kaggle/small_track/` | `kernel-metadata.json` for the training kernel (notebook copied in at push time, gitignored) |
| `kaggle/net_probe/` | scratch kernel used to diagnose Kaggle platform quirks (gitignored, not part of the pipeline) |
| `scripts/kaggle_run.py` | push + wait + pull one training run |
| `scripts/kaggle_status.py` | readiness check (`make kaggle-status`) |
| `eurosat_cache/` | local copy of the built training-data cache, before/after publishing as a dataset (gitignored) |
| `kaggle_runs/<timestamp>/` | pulled results of each training run (gitignored) |
