# Project Status

## Last Updated

July 26, 2026

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
  Updated July 22, 2026: `baseline_inverse_config` is now filled in with the
  real original `heat_pinn_basic.ipynb` inverse hyperparameters (no longer
  `None` placeholders), and both the forward and inverse sections of
  `heat_pinn_tuned.ipynb` now run their baseline configs through the shared
  training functions. What's left before `heat_pinn_basic.ipynb` can actually
  be removed: nothing has been *run* yet this session beyond tiny-config
  smoke tests -- a real top-to-bottom notebook run (or at least the baseline
  cells) is needed to confirm both baseline numbers look sane before treating
  `heat_pinn_basic.ipynb` as safely retirable.
- Found July 22, 2026 during a full read-through of the reorganized
  notebook: the inverse Optuna sweep's trial-selection objective and the
  final retrain loop's reported "tuned inverse PINN" accuracy both used the
  identical, single fixed noisy observation draw. Fixed July 26, 2026 (see
  "Completed Cleanup Work") by splitting the inverse pipeline into a search
  stage (unchanged, one fixed dataset), a confirmation round (picks the
  winning config on fresh datasets the search stage never saw), and a final
  evaluation (retrains only the frozen winner on datasets untouched by
  either earlier stage, and this is what gets reported and compared against
  CN-NLS) -- so no single dataset both selects a winner and reports its
  accuracy.
- New July 22, 2026: baseline PINN field-evaluation time is never measured
  for the forward problem. `CLAUDE.md` requires reporting PINN
  field-evaluation time as its own cost line; the tuned model's is measured
  (`pinn_infer_time`) and shown in the comparison table, but baseline's
  field evaluation happens (inside the test-error loop) without ever being
  timed, so its row in the final table is `N/A` where it should be a number.
- New July 22, 2026: the forward and inverse Optuna search spaces differ in
  undocumented ways -- inverse hardcodes `activation="sin"` and `N_f=10000`
  while forward searches both; inverse's `N_bc`/`N_ic` search ranges are
  exactly double forward's. Most notably, forward's own completed HPO run
  shows its best trials all preferring `tanh`, but inverse never lets Optuna
  try `tanh` at all -- if tanh is genuinely better for this PDE, inverse may
  be leaving accuracy on the table for no documented reason. Needs a
  deliberate decision (keep as an intentional prior and document why, or
  widen inverse's search space to match), not a silent fix, since it changes
  real Optuna search behavior.
- New July 22, 2026: the forward problem's true diffusivity (0.4) is a bare,
  repeated literal with no single source of truth -- hardcoded separately in
  `pinn_shared.py`'s `train_forward` (which does not read it from
  `forward_config` at all), in `heat_pinn_tuned.ipynb`'s grid cell, and again
  in the final FD-vs-PINN comparison cell. Compare to inverse, where
  `true_alpha` lives once in `INVERSE_FIXED_CONFIG`. Fixing this means
  changing `train_forward`'s signature in `pinn_shared.py`, so it's a real
  change requiring approval, not a trivial one.
- Minor, low-priority cosmetic-only inconsistencies noticed July 22, 2026
  (not correctness issues): `FORWARD_FIXED_CONFIG["lambda_pde"]` is an int
  (`1`) while `INVERSE_FIXED_CONFIG["lambda_pde"]` is a float (`1.0`),
  functionally identical; forward's mean+/-std reporting variables use a
  `pinn_` naming prefix while inverse's use `tuned_inverse_` for the
  analogous quantity; a comment in `heat_pinn_tuned.ipynb`'s representative-
  run plotting cell references an "avoid two copies" dedup rationale that
  is now slightly stale phrasing since the grid is built once from the start
  rather than deduplicated after the fact.
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

## Completed Cleanup Work (July 21, 2026)

- Added a "baseline (no-tuning) forward PINN" cell to `heat_pinn_tuned.ipynb`
  that runs the pre-existing (previously unused) `forward_config` -- the
  literature-informed hyperparameters originally used in
  `heat_pinn_basic.ipynb` -- across 5 seeds, reporting mean +/- std rel L2
  error on both the validation grid and the held-out test grid, using the
  same multi-seed methodology as the tuned top-configs retrain loop.
- Added a `baseline_inverse_config` scaffold and matching multi-seed
  training loop for the inverse problem; all hyperparameter values are
  `None` placeholders pending the researcher transcribing the exact
  original `heat_pinn_basic.ipynb` inverse config.
- Moved `compute_loss` (forward-problem loss) and `train_forward` from
  `heat_pinn_tuned.ipynb` into `pinn_shared.py`, verified byte-for-byte
  identical to the notebook originals except for the same local
  `import optuna` fix already applied to `train_inverse`. `pinn_shared.py`
  now holds the complete shared "algorithm layer" (architecture, sampling,
  losses, training loops, FD solver, CN-NLS) for both the forward and
  inverse problems.
- Removed a duplicate held-out test-grid construction in the
  single-representative-run plotting cell; it now reuses the grid already
  built in the retrain-loop cell instead of rebuilding an identical one via
  a separate `linspace`/`meshgrid` call.

Testing for all of the above was quick standalone script checks: tiny-config
smoke tests of the new baseline loops run outside the notebook, a diff
confirming the moved functions are byte-identical to the notebook originals
aside from the documented `import optuna` addition, and a numerical check
that the deduplicated test grid produces bit-for-bit identical values to the
old duplicated construction.

## Completed Cleanup Work (July 22, 2026)

All changes are in `heat_pinn_tuned.ipynb`; `pinn_shared.py` was not touched
this session.

- Filled in `baseline_inverse_config` with the real original
  `heat_pinn_basic.ipynb` inverse hyperparameters (previously `None`
  placeholders).
- `top_configs` (forward) and a new `top_inverse_configs` (inverse) now pull
  retrain candidates dynamically from the completed Optuna study's
  `sorted_trials` instead of hand-transcribed numbers, via
  `FORWARD_FIXED_CONFIG`/`INVERSE_FIXED_CONFIG` dicts that each define their
  problem's non-searched hyperparameters exactly once (previously
  duplicated between the Optuna objective and the retrain loop).
- Reordered both the forward and inverse sections into the same methodology
  order: baseline PINN -> Optuna-tuned PINN -> comparison against the
  classical method.
- Built the inverse section's missing multi-seed retrain/selection step
  (mirroring the forward one), fixing a bug where the reported "tuned
  inverse PINN" alpha-error and its illustrative loss-curve plot both came
  from a disconnected, single-seed, hand-set `inverse_config` that was never
  reconciled with what Optuna actually found. `inverse_config` is retired
  entirely -- nothing needs it once the retrain loop and CN-NLS call are
  repointed at real values.
- Extended both sections' final comparison cells into 3-column
  Baseline/Tuned/Classical-method tables (forward: PINN vs. FD; inverse:
  PINN vs. CN-NLS), and added Optuna total search time as its own reported
  cost line for both problems (previously not tracked at all).
- Repointed the CN-NLS comparison at the explicit shared inverse-Optuna
  observations instead of values that happened to be numerically identical
  by coincidence (same seed, same shape) from a one-off demo cell.

Testing was quick standalone script checks: dict-merge and
winner/representative-selection logic exercised with tiny configs and fake
Optuna trial objects; a grep pass confirming no stray references to the
retired `inverse_config` or the old observation-access pattern remain; and a
full manual trace confirming every cell's variable references are satisfied
by something above it in the new order. No real Optuna sweep or full
notebook execution was run.

A follow-up review pass (see the "Known Issues to Investigate" entries
above) found several methodology questions and one hardcoded-value
redundancy that predate this session's reordering but were surfaced by it;
none have been fixed yet (the observation-reuse item was fixed July 26,
2026 -- see below).

## Completed Cleanup Work (July 26, 2026)

All changes are in `heat_pinn_tuned.ipynb`; `pinn_shared.py` was not touched
this session.

- Fixed the inverse Optuna observation-reuse issue flagged July 22, 2026:
  split the inverse pipeline into three stages using non-overlapping seed
  ranges -- search (seed 0, unchanged, used only by `objective_inverse`),
  confirmation round (seeds 1-5, retrains Optuna's top 3 configs on fresh
  datasets to pick a winner without reporting any accuracy number from this
  stage), and final evaluation (seeds 6-10, retrains only the frozen winner
  on datasets untouched by search or confirmation, producing the mean +/-
  std that's actually reported). CN-NLS in the final comparison now runs
  once per those same 5 final-evaluation datasets instead of once against
  the search-stage dataset, so PINN and CN-NLS are compared on identical,
  never-used-for-selection data, and CN-NLS gets its own mean +/- std
  instead of a single-run number.
- Added `INVERSE_OBS_CONFIG` (`N_obs`/`noise_std`/`true_alpha`) as a single
  source of truth for observation generation across all three stages,
  ahead of planned follow-up sensitivity sweeps (varying noise level at
  fixed observation count, and varying observation count at fixed noise
  level, for both the tuned PINN and CN-NLS).
- Removed a dead `alpha = 0.4` variable in the forward retrain-loop cell,
  left over from before the July 21 grid-deduplication fix; confirmed via
  grep it was never read anywhere else.
- Fixed the inverse comparison table's row labels (`"(5 seeds)"` ->
  `"(5 runs)"`): "seeds" implied only training randomness varies between
  runs, which stopped being true for the Tuned and CN-NLS columns once each
  of their 5 runs began using an independently-drawn dataset rather than a
  shared one.

Testing was a standalone smoke test (tiny configs, fake Optuna trial
objects) exercising all three inverse stages end-to-end, with an explicit
check that the final-evaluation stage's 5 datasets are numerically distinct
from each other; a full read-through plus grep passes confirming no stale
references to pre-split variable names and no leftover use of the
search-stage observations outside the search cell; and a JSON-validity
check after the two small cleanup edits. No real Optuna sweep or full
notebook execution was run.

## Current Uncommitted Changes

None, pending this session's commit (see "Completed Cleanup Work (July 22,
2026)" and "(July 26, 2026)" above). `methodology-cleanup` will be ahead of
`origin/methodology-cleanup` and not yet pushed.

## Next Recommended Step

Pipeline inspection, reordering, and the observation-reuse fix are
complete; pick up the next item on the priority-ordered backlog (ordered
biggest-impact first, per researcher preference):

1. Decide on the forward/inverse Optuna search-space asymmetry (notably
   `activation` fixed to `"sin"` for inverse despite forward's own best
   trials preferring `"tanh"`) -- a methodology call, not code cleanup (see
   "Known Issues to Investigate").
2. Fix inverse Optuna pruning comparing weighted loss across trials with
   different loss weights.
3. Add baseline PINN field-evaluation timing for forward, and consolidate
   the forward problem's hardcoded true-diffusivity literal (0.4) the same
   way `INVERSE_FIXED_CONFIG["true_alpha"]` already does for inverse (touches
   `train_forward`'s signature in `pinn_shared.py`).
4. Build the planned sensitivity sweeps -- vary `noise_std` at fixed
   `N_obs`, and vary `N_obs` at fixed `noise_std` -- comparing the tuned
   PINN and CN-NLS at each point. `INVERSE_OBS_CONFIG` is already factored
   out specifically to support this.
5. Once the above are settled: actually run the real Optuna sweeps and
   retrain loops (a real, approval-gated training job) to get real baseline/
   tuned numbers, confirm they look sane, and only then remove
   `heat_pinn_basic.ipynb`.
6. Smaller polish items: `N_bc` naming clarification, `requirements.txt`,
   fleshing out `README.md`, and the minor cosmetic-only naming/comment
   inconsistencies noted above.

Revisit the deferred inverse-alpha positivity question (see "Known Issues to
Investigate") before final results are reported.

## Suggested First Message to Claude

Read `CLAUDE.md` and `PROJECT_STATUS.md`, check the current Git branch and Git
status, and review the most recent commits. Explain where the project stands
and propose the next cleanup step from "Next Recommended Step" above. Do not
edit anything yet.