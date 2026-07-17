# Project Status

## Last Updated

July 17, 2026

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
- `inverse_hpo_study.pkl`

No methodological cleanup has begun yet.

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

- The Crank-Nicolson solver currently uses a dense matrix solve even though its
  matrix is tridiagonal.
- The CN matrix is refactored at every timestep instead of once per diffusivity.
- The CN solver calculates analytical error internally even when that error is
  discarded by the inverse loop.
- The inverse PINN generates new noisy observations inside each training call.
- Different Optuna trials can therefore be evaluated on different datasets.
- Observation generation, collocation sampling, and model initialization use
  shared global random-number state.
- The inverse PINN diffusivity is not constrained to remain positive or within
  the CN search range.
- Optuna pruning uses weighted training loss even though the loss weights vary
  between trials.
- The forward Optuna objective uses the same analytical grid later treated as a
  final test grid.
- The forward notebook retains the luckiest model among multiple configurations
  and seeds.
- PINN and CN timing boundaries currently measure different kinds of work.
- `N_bc` represents points per boundary, so the actual total is twice the
  configuration value.
- The CN function accepts `L` and `T` but still hardcodes unit-domain grids.
- The inverse CN candidate count is hardcoded even though the configuration
  contains a candidate-count setting.
- The notebook does not currently run cleanly from top to bottom.

These are investigation targets, not permission to change everything at once.

## Completed Setup Work

- Created a private GitHub repository.
- Added a project-specific `.gitignore`.
- Preserved the original work on `main`.
- Created the `methodology-cleanup` branch.
- Added the current source files and notebooks.
- Added persistent Claude project instructions.
- Added this session-handoff file.

## Current Uncommitted Changes

`CLAUDE.md` and `PROJECT_STATUS.md` need to be added, reviewed, committed, and
pushed.

## Next Recommended Step

Inspect the complete current pipeline without editing it.

Specifically:

1. Trace forward PINN training and evaluation.
2. Trace inverse PINN training and observation generation.
3. Trace the Crank-Nicolson forward solver.
4. Trace the CN-based inverse estimator.
5. Trace both Optuna objectives.
6. Identify which functions live in `pinn_shared.py` and which remain in the
   notebooks.
7. Explain the pipeline in plain language.
8. Propose only the first small cleanup change.
9. Wait for researcher approval before editing.

## Suggested First Message to Claude

Read `CLAUDE.md` and `PROJECT_STATUS.md`, check the current Git branch and Git
status, and inspect the current project pipeline. Explain where the project
stands and propose only the first small cleanup step. Do not edit anything yet.