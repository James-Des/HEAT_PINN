# Project Status

## Last Updated

July 20, 2026

## Current Branch

`methodology-cleanup`

The original project is preserved on the `main` branch.

## Current Project State

The existing PINN project has been copied into a Git repository and pushed to
GitHub.

The repository currently contains:

- `pinn_shared.py`
- `heat_pinn_basic.ipynb`
- `heat_pinn_tuned.ipynb`
- `James_Desjarlais_PINN_Final.pdf`
- `README.md`
- `.gitignore`
- `CLAUDE.md`
- `PROJECT_STATUS.md`

The following generated Optuna files are available locally but ignored by Git:

- `optuna_forward.db`
- `forward_hpo_study.pkl`
- `optuna_inverse.db`
- `inverse_hpo_study.pkl`

Methodological cleanup began July 20, 2026 (see "Completed Cleanup Work"
below). All Optuna study files above still reflect the pre-cleanup
methodology (leaky validation grid for forward HPO, non-reproducible
per-trial observations for inverse HPO) until a fresh sweep is explicitly
approved and run.

## Research Structure

The final study should compare:

1. An efficient Crank-Nicolson forward solver.
2. A literature-informed baseline PINN.
3. An Optuna-tuned forward PINN.
4. A baseline inverse PINN.
5. An Optuna-tuned inverse PINN.
6. A CN-based least-squares inverse estimator.

The paper should report both final-model performance and the computational cost
required to obtain that performance.

## Known Issues to Investigate

These issues have been identified but not yet corrected:

- Observation generation, collocation sampling, and model initialization use
  shared global random-number state. Observation generation specifically was
  fixed July 20, 2026 (see "Completed Cleanup Work"); collocation sampling and
  model init still share global RNG state, which is normal/acceptable
  training-procedure stochasticity rather than a fairness bug, per July 20
  discussion.
- The inverse PINN diffusivity is not constrained to remain positive or within
  the CN search range. Discussed and intentionally deferred July 20, 2026: a
  `log(alpha)` reparameterization was proposed and rejected as too large a
  change to how the central estimated quantity is optimized; a lighter
  clamp-based safety net was also proposed and deferred. Revisit before final
  results are reported.
- Optuna pruning uses weighted training loss even though the loss weights vary
  between trials.
- `N_bc` represents points per boundary, so the actual total is twice the
  configuration value.
- `heat_pinn_basic.ipynb` does not currently run cleanly from top to bottom and
  duplicates (with some divergence) logic that now lives in `pinn_shared.py`.
  Plan: extract its baseline configs into one final consolidated notebook that
  imports from `pinn_shared.py`, verify/save corrected baseline results, then
  remove the duplicate notebook from `methodology-cleanup` (preserved on
  `main`). Do not fix it in place.
- No `requirements.txt`/environment file, and `README.md` is a single
  placeholder sentence -- a fresh clone currently has no setup instructions or
  dependency list.

These are investigation targets, not permission to change everything at once.

## Completed Setup Work

- Created a private GitHub repository.
- Added a project-specific `.gitignore`.
- Preserved the original work on `main`.
- Created the `methodology-cleanup` branch.
- Added the current source files and notebooks.
- Added persistent Claude project instructions.
- Added this session-handoff file.

## Completed Cleanup Work (July 20, 2026)

All changes are in `pinn_shared.py` and `heat_pinn_tuned.ipynb` only;
`heat_pinn_basic.ipynb` was intentionally left untouched (see "Known Issues
to Investigate").

- Replaced `fd_solver`'s dense per-timestep `np.linalg.solve` with a sparse
  tridiagonal matrix factored once and reused across all timesteps (~385x
  speedup at a 1000x1000 grid, numerically identical results).
- Added an optional `compute_error` flag to `fd_solver` so callers that
  discard the analytical error (e.g. `cn_nls_baseline`'s 200-candidate loop)
  don't pay for computing it.
- `cn_nls_baseline` now returns `cn_nls_time` instead of only printing it,
  and reads `inverse_config["n_alpha_candidates"]` instead of hardcoding 200.
- Removed `fd_solver`'s non-functional `L`/`T` parameters (they only
  half-affected behavior -- `dx`/`dt` but not the grid/IC/exact-solution
  formulas) and hardcoded the unit domain the experiment always uses.
- Tuned notebook's FD-vs-PINN comparison cell: `fd_time` no longer includes
  analytical-error computation, matching how `pinn_infer_time` is measured.
- `train_inverse` now takes `x_obs`/`t_obs`/`u_obs` as required arguments
  instead of generating them internally, so Optuna trials and repeated runs
  are compared on identical observations instead of a fresh random draw
  each call.
- `train_forward`'s internal evaluation grid (used for the HPO objective and
  retrain-loop model selection) is now offset from the grid used for final
  reporting, so the same points can no longer both select and evaluate the
  winning model (train/test leakage fix).
- Tuned notebook's top-configs retrain loop now selects the winning config
  by mean validation error across 5 seeds (not any single run's error),
  reports mean +/- std of held-out test error as the number to quote as PINN
  accuracy, and keeps only a closest-to-mean representative run for the
  plots (previously kept whichever single run had the best individual
  error).
- Fixed an uncommitted typo in `cn_nls_baseline` (`how_history` vs
  `mse_history`) that would have raised a `NameError`.

Testing for all of the above was quick standalone script checks (regression
comparisons against pre-change behavior, small mock Optuna sweeps,
reproducibility checks) rather than full notebook execution or real Optuna
sweeps.

## Current Uncommitted Changes

None. Working tree is clean as of July 20, 2026. `methodology-cleanup` is
several commits ahead of `origin/methodology-cleanup` and has not been
pushed.

## Next Recommended Step

Pipeline inspection is complete; pick up the next item on the
priority-ordered backlog (ordered biggest-impact first, per researcher
preference):

1. Fix inverse Optuna pruning comparing weighted loss across trials with
   different loss weights, or
2. Begin the `heat_pinn_basic.ipynb` consolidation (extract baseline configs
   into a final notebook, retire the duplicate), or
3. Smaller polish items: `N_bc` naming clarification, `requirements.txt` and
   fleshing out `README.md`.

Revisit the deferred inverse-alpha positivity question (see "Known Issues to
Investigate") before final results are reported.

## Suggested First Message to Claude

Read `CLAUDE.md` and `PROJECT_STATUS.md`, check the current Git branch and Git
status, and review the most recent commits. Explain where the project stands
and propose the next cleanup step from "Next Recommended Step" above. Do not
edit anything yet.