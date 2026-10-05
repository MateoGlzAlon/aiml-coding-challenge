# Course material (HSG-AIML-Teaching/ML2026-Lab) is a git submodule in ML2026-Lab.
.DEFAULT_GOAL := help
.PHONY: help setup update

help: ## List the available targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'

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
