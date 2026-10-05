# Course material (HSG-AIML-Teaching/ML2026-Lab) is a git submodule in ML2026-Lab.
.DEFAULT_GOAL := help
.PHONY: help setup update kaggle-venv kaggle-status kaggle-cache-build kaggle-cache-wait kaggle-cache-download kaggle-cache-dataset kaggle-cache submit

help: ## List the available targets
	@grep -E '^[a-z][a-z0-9-]*:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-20s %s\n", $$1, $$2}'

setup: ## Fetch the course material at the commit this repo records (after a fresh clone)
	git submodule sync ML2026-Lab
	git submodule update --init ML2026-Lab

update: ## Pull the latest course material and stage the new submodule commit
	@old=$$(git ls-files -s ML2026-Lab | cut -d' ' -f2); \
	git submodule update --init --remote ML2026-Lab || exit 1; \
	new=$$(git -C ML2026-Lab rev-parse HEAD); \
	if [ "$$old" = "$$new" ]; then \
		echo "Already up to date: $$(git -C ML2026-Lab log -1 --format='%h %s (%cr)')"; \
	else \
		echo "New course commits:"; \
		git -C ML2026-Lab log --oneline $$old..$$new; \
		echo "Folders changed: $$(git -C ML2026-Lab diff --name-only $$old $$new | cut -d/ -f1 | sort -u | tr '\n' ' ')"; \
		git add ML2026-Lab; \
		echo "Staged ML2026-Lab. Commit it with: git commit -m 'Update course material'"; \
	fi

# Kaggle automation (see KAGGLE_WORKFLOW.md): the training kernel needs the EuroSAT MS
# training set pre-decoded and attached as a read-only Kaggle Dataset, instead of
# re-downloading + rasterio-decoding 27,000 GeoTIFFs from Zenodo on every run. These
# targets build that cache once, on a Kaggle kernel (not locally - the decode step needs
# more RAM than this repo's laptops reliably have free), and publish it as a dataset.
CACHE_KERNEL := mateogonzalezalonso/eurosat-ms-cache-builder
CACHE_DATASET := mateogonzalezalonso/eurosat-ms-l1c-cache
CACHE_DIR := eurosat_cache

kaggle-venv: ## Create .venv and install the Kaggle CLI (needs ~/.kaggle/access_token first, see KAGGLE_WORKFLOW.md)
	test -d .venv || python3 -m venv .venv
	.venv/bin/pip install -q --upgrade kaggle

kaggle-status: kaggle-venv ## Check whether auth, competition access, and the training-data cache are ready
	@.venv/bin/python scripts/kaggle_status.py

kaggle-cache-build: kaggle-venv ## Push the cache-build kernel (downloads EuroSAT_MS.zip and decodes it, on Kaggle)
	.venv/bin/kaggle kernels push -p kaggle/build_cache

kaggle-cache-wait: kaggle-venv ## Poll the cache-build kernel until it finishes (COMPLETE/ERROR)
	@until .venv/bin/kaggle kernels status $(CACHE_KERNEL) 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 30; done
	.venv/bin/kaggle kernels status $(CACHE_KERNEL)

kaggle-cache-download: kaggleLet me directly re-inspect the exact current cell 4 content in the real notebook, since the isolated probe works fine but the real notebook doesn't — something must differ.-venv ## Download the finished cache-build kernel's output into eurosat_cache/
	mkdir -p $(CACHE_DIR)
	.venv/bin/kaggle kernels output $(CACHE_KERNEL) -p $(CACHE_DIR)

kaggle-cache-dataset: kaggle-venv ## Publish eurosat_cache/*.npy as the private dataset the training kernel attaches
	cp $(CACHE_DIR)/X_ms.npy $(CACHE_DIR)/y.npy $(CACHE_DIR)/classes.json kaggle/cache_dataset/
	@if .venv/bin/kaggle datasets status $(CACHE_DATASET) >/dev/null 2>&1; then \
		.venv/bin/kaggle datasets version -p kaggle/cache_dataset -m "rebuild $$(date +%Y-%m-%d)" -d; \
	else \
		.venv/bin/kaggle datasets create -p kaggle/cache_dataset -r zip; \
	fi

kaggle-cache: kaggle-cache-build kaggle-cache-wait kaggle-cache-download kaggle-cache-dataset ## Build + download + publish the training-data cache end to end (one-time)

submit:
	.venv/bin/python scripts/kaggle_run.py --submit
