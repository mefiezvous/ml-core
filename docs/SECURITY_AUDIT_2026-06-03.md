# Security Audit — ml-core
**Date**: 2026-06-03
**Auditor**: SEC-MLcore (Claude opus 4.7, Plan agent)
**Scope**: code (HEAD), CI workflows, deps lockfile, git history

## Executive summary

ml-core is a public Apache-2.0 ML library with a small attack surface (no network listeners, no shell-out, no eval/exec). The repo is in solid shape overall, but a **High-severity RCE risk** exists in `CheckpointManager.load_latest()` which calls `torch.load(path, weights_only=False)` — combined with the optional pull of checkpoints from HuggingFace Hub, this is the canonical "pickle deposited on the Hub" RCE vector. Secondary issues: missing `LICENSE` file (README link broken) despite Apache-2.0 claim, missing `SECURITY.md`, missing `.pre-commit-config.yaml` (CONTRIBUTING.md falsely advertises pre-commit hooks), CI does not pin third-party actions to SHA, no Dependabot, no CodeQL, and CI lacks an explicit minimum `permissions:` block (defaults to write on classic repos). No secrets found in git history, no dangerous `eval`/`exec`/`subprocess`/`yaml.load`/`pickle.load` calls outside `torch.load`.

## Findings

| ID | Severity | Surface | Finding | Recommended fix | Effort |
|----|----------|---------|---------|-----------------|--------|
| MLC-001 | High | Pickle | `src/mlcore/training/checkpoint_manager.py:61` calls `torch.load(path, weights_only=False)`. The `local_dir` can hold checkpoints downloaded/synced from the HuggingFace Hub repo configured via `hf_repo_id`. A malicious actor with push rights to that Hub repo (or a typosquat / supply-chain compromise) can ship a pickle payload that achieves arbitrary code execution on `Trainer.__init__` (which calls `_resume_from_checkpoint` → `ckpt_manager.load_latest`). torch 2.10 defaults to `weights_only=True`; the explicit `False` opts out of the safety net. | Set `weights_only=True`. If you genuinely need to load non-tensor state (e.g. optimiser steps as Python ints), restrict with `torch.serialization.add_safe_globals([...])` and keep `weights_only=True`. Add a unit test that confirms a crafted pickle is rejected. | S |
| MLC-002 | Medium | Pickle | `src/mlcore/policies/act_wrapper.py:110` and `src/mlcore/policies/diffusion_wrapper.py:102` call `torch.load(path, map_location=self._device)` without `weights_only=`. On torch 2.10 the default is `True`, so this is currently safe — but the project pins `lerobot==0.5.1` only; if a future env downgrades torch (<2.6) or a developer runs on an older interpreter, the call silently becomes unsafe. | Make the safe behaviour explicit: pass `weights_only=True` in both wrappers. Same residual whitelisting strategy as MLC-001 if needed. | S |
| MLC-003 | High | Pre-commit / Docs | `docs/CONTRIBUTING.md` line 33 states: *"Pre-commit hooks (ruff, mypy, anti-leak) run on every commit."* But there is **no `.pre-commit-config.yaml`** in the repo. The advertised "anti-leak" hook does not exist, giving contributors a false sense of secret-detection coverage on a public repo. | Add `.pre-commit-config.yaml` wiring `ruff`, `ruff-format`, `mypy`, plus a secret-detection hook (`gitleaks` or `detect-secrets`) and `python-safety-dependencies-check`. Mention `pre-commit install` in the README. Either way, align CONTRIBUTING.md with reality. | M |
| MLC-004 | Medium | CI | `.github/workflows/ci.yaml` uses `actions/checkout@v4` and `astral-sh/setup-uv@v5` pinned by **tag, not SHA**. A tag-mutating supply-chain attack against either action would silently execute attacker code on every PR with access to whatever the default `GITHUB_TOKEN` permissions are. | Pin to commit SHA: e.g. `actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11 # v4.1.1` (and similarly for `setup-uv`). Pair this with Dependabot (`.github/dependabot.yml`) configured for `package-ecosystem: github-actions` so SHAs are kept current automatically. | S |
| MLC-005 | Medium | CI | The workflow declares no top-level `permissions:` block. On classic GitHub repos the default `GITHUB_TOKEN` grants `contents: write` and broader scopes. A compromised action (see MLC-004) or a poisoned dependency could push to `main` or open malicious releases. | Add at workflow root: `permissions: { contents: read }`. Grant per-job elevated scope only where strictly needed (none of the current jobs need write). Also set `concurrency:` to avoid duplicate runs and consider `pull_request_target` discipline if added later. | S |
| MLC-006 | Medium | IP / Docs | `README.md` line 76 links to `[LICENSE](LICENSE)` and `pyproject.toml` declares `license = { text = "Apache-2.0" }`, but **no `LICENSE` file is tracked** in git. The repo is public Apache-2.0 in intent but lacks the canonical legal text required by the Apache-2.0 redistribution clauses. Downstream consumers cannot satisfy clause 4(c) (must include a copy of the license). | Commit the canonical Apache-2.0 license text as `LICENSE` at repo root. Optionally add `NOTICE` per clause 4(d) if any third-party attribution is desired. | S |
| MLC-007 | Medium | Docs | No `SECURITY.md` at repo root. GitHub displays a "Security policy" tab pointing nowhere; researchers have no documented disclosure channel for a public repo. | Add `SECURITY.md` with: (a) supported versions matrix, (b) reporting address (private channel — GitHub Security Advisories preferred over email), (c) expected response SLA. Link from README. | S |
| MLC-008 | Low | Network / Secrets | `src/mlcore/training/trainer.py:98` reads `os.environ.get("WANDB_API_KEY")`. This is a soft trip-wire — used only as a *presence* check, never logged or transmitted by ml-core itself (wandb SDK handles the network call). Also violates the project's own rule "No direct `os.environ` access" stated in CLAUDE.md line 6 and CONTRIBUTING.md line 41. Consistency, not security, but worth fixing. | Move the env probe to a Hydra cfg flag (`logging.wandb_enabled`) or expose an explicit constructor argument. Document that the HF Hub token (`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`) is consumed implicitly by `huggingface_hub.upload_file`; never logged. | S |
| MLC-009 | Low | Validation | `src/mlcore/robots/base.py` uses a plain frozen `@dataclass` for `RobotSpec` with no field-level validation (e.g. `n_joints > 0`, `action_dim > 0`, `0 < success_threshold < 1`, non-empty `obs_keys`). `validate_spec_against_env()` only cross-checks against an env. A malformed spec registered via the public `register()` would produce confusing downstream errors but no security impact. | Add `__post_init__` invariants (raise `ValueError` on negative dims, empty `obs_keys`, etc.). Document that the global `_REGISTRY` is not thread-safe (it's a module-level dict with no lock — registering from multiple threads is unsafe but currently only used at import time). | S |
| MLC-010 | Low | CI / Deps | No `pip-audit` / `safety` / `osv-scanner` step in CI. Lockfile (`uv.lock`) pins ~all deps, but nothing alerts on a newly published CVE against torch / mlflow / transformers / pillow / cryptography / etc. The repo also has no `.github/dependabot.yml`. | Add a CI job (or weekly scheduled workflow) running `uv pip install pip-audit && uv run pip-audit --strict` against the lockfile, or `osv-scanner -r .`. Add `.github/dependabot.yml` for `pip`, `github-actions`. | S |
| MLC-011 | Info | SPDX / IP | SPDX header coverage **100%** on tracked Python files (31/31 in `src/`, 18/18 in `tests/`) — header rule from CLAUDE.md respected. Also confirmed: no `LicenseRef-Proprietary`, no "All Rights Reserved" in tree. | Maintain via a `reuse lint` (or `pre-commit-hook-spdx`) hook once MLC-003 is implemented. | S |
| MLC-012 | Info | Secrets | Git history scan (`git log -p --all -S "TOKEN" -S "SECRET" -S "password" -S "hf_"`) returned **no leaked credential**. The only `hf_` hit was inside a commit message body (refactor description), not a real token. No `.env`, no `credentials.json`, no `*.key` tracked. `.gitignore` covers build/venv artefacts. | None — keep this clean by enforcing MLC-003. | — |
| MLC-013 | Info | Network | Static review of `src/mlcore/` confirms **no listening sockets, no `requests.*`, no `urllib`, no `subprocess`, no `os.system`, no `shell=True`**. All outbound network is delegated to `huggingface_hub` (Hub push) and `mlflow`/`wandb` SDKs. Surface matches the documented design. | None. Re-verify if `requests` or a webhook is ever added. | — |

Severity = Critical / High / Medium / Low / Info
Effort = S (<1h) / M (1-4h) / L (>4h)

## Out of scope

- Hardware/HAL layer — lives in `robotics-platform-template`, audited separately.
- Downstream consumers (`lerobot-playground-portfolio`, etc.).
- MLflow server hardening — ml-core only uses local `mlruns/` URIs; remote tracking server config is a deployer concern.
- HF Hub repo permissions and branch protection on `mefiezvous/ml-core` itself — those are GitHub UI settings, not code.
- Runtime supply-chain scanning of compiled `torch` / `mujoco` wheels — out of scope; covered by upstream wheel signing if any.
- `lerobot` upstream code that gets imported and exercised — its own security boundary.

## Recommandations transverses

- **Treat checkpoints as untrusted code**, not data. Any path that flows from Hub → local disk → `torch.load` must be `weights_only=True`, OR run inside a sandboxed worker. MLC-001 is the single most valuable fix in this audit; it costs ~1 line and removes a plausible RCE for any consumer of the library.
- **Tighten the CI supply chain in one PR**: SHA-pin actions (MLC-004), add top-level `permissions: read` (MLC-005), wire `pip-audit` + Dependabot (MLC-010). Three small changes that together raise the bar against the most common GitHub-Actions-driven attacks.
- **Make the public-repo paperwork match reality**: ship `LICENSE` (MLC-006), `SECURITY.md` (MLC-007), and `.pre-commit-config.yaml` with a secret-detection hook (MLC-003) — and update CONTRIBUTING.md to stop advertising hooks that do not exist.

### Critical files for implementation (fixing findings)

- `src/mlcore/training/checkpoint_manager.py` (MLC-001 — line 61)
- `src/mlcore/policies/act_wrapper.py` (MLC-002 — line 110)
- `src/mlcore/policies/diffusion_wrapper.py` (MLC-002 — line 102)
- `.github/workflows/ci.yaml` (MLC-004, MLC-005, MLC-010)
- `docs/CONTRIBUTING.md` (MLC-003 — line 33 false claim)
- `LICENSE` (MLC-006 — to be created)
- `SECURITY.md` (MLC-007 — to be created)
- `.pre-commit-config.yaml` (MLC-003 — to be created)
- `.github/dependabot.yml` (MLC-010 — to be created)
