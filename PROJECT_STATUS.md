# Project Status

## Last Updated

August 21, 2026

## Current Branch

`methodology-cleanup`

The original project is preserved on the `main` branch.

## Hardware and Environment

Development has moved from a CPU-only MacBook Pro to a new Windows PC with an
NVIDIA RTX 5070 Ti (CUDA-enabled). Python 3.12 and a fresh virtual environment
(`.venv`, gitignored) were set up on this machine on August 21, 2026, with all
required packages installed, including CUDA-enabled `torch 2.11.0+cu128`
(GPU detection confirmed working). Git for Windows was also installed the same
day -- only GitHub Desktop (with its own bundled, non-PATH git) was present
before, so command-line git now works directly in a terminal/VSCode after a
restart.

Prior MacBook-generated results will not be reused. All real Optuna sweeps
and final comparisons will be run fresh on this machine so every reported
cost line shares one consistent hardware baseline.

## GPU Migration Action Items

Agreed August 21, 2026, sequenced as separate small changes (each gets its
own diff and test before moving to the next, per the Researcher Learning
Requirement in `CLAUDE.md`). These are a prerequisite for item 5 ("run the
real Optuna sweeps") in "Next Recommended Step" below -- no point running
real sweeps until training actually uses the GPU.

1. Add a `device` parameter to `train_forward`/`train_inverse` in
   `pinn_shared.py`. Defaults to auto-detect (GPU if available, else CPU);
   explicit `device="cpu"` reserved for the control-timing run. Moves the
   model, sampled points, and (for inverse) `alpha`/observations onto the
   resolved device. -- DONE AND VERIFIED (August 21, 2026). The initial
   commit only updated the inside of these two functions; two call sites
   still assumed CPU tensors and would have crashed on this GPU machine:
   `cn_nls_baseline` called `.numpy()` directly on `x_obs`/`t_obs`/`u_obs`
   (now GPU tensors when returned from `train_inverse`), and three cells in
   `heat_pinn_tuned.ipynb` called a GPU-resident model on the CPU-resident
   shared test grid. Both fixed: `cn_nls_baseline` now uses
   `.cpu().numpy()`, and the three notebook cells move the test grid to
   *that specific model's* actual device (via
   `next(model.parameters()).device`, not a blanket notebook-level
   variable) and bring the prediction back with `.cpu()` right after --
   this specifically preserves the explicit `device="cpu"` control-timing
   path above, which a hardcoded notebook-level device variable would have
   broken. Verified via standalone scratch scripts covering the default-GPU
   path, the explicit-CPU path, and `cn_nls_baseline` against GPU-resident
   observations.
2. Remove the per-iteration `.item()` GPU sync in both training loops.
   Append `total_loss.detach()` during the loop instead of calling `.item()`
   every step; convert to floats once at the end via
   `torch.stack(...).cpu().tolist()`. The periodic print and Optuna-pruning
   `.item()` calls (every 200 iters) stay as-is. -- DONE AND VERIFIED
   (August 21, 2026). One subtlety found during implementation:
   `train_inverse`'s `alpha` is a single `nn.Parameter` the optimizer
   mutates in place every step, so `alpha.detach()` alone would make every
   stored history entry alias the same storage and collapse the whole
   alpha-convergence history to one repeated final value; fixed with
   `alpha.detach().clone()` instead. The loss tensors are fresh objects
   each iteration (not mutated in place), so plain `.detach()` is correct
   for those. Verified via a scratch script confirming `history_alpha`
   contains genuine per-iteration variation (319/320 unique values across
   320 iterations) rather than a collapsed repeated value.
3. Set `torch.backends.cudnn.deterministic = True` inside `set_seed()` so
   reproducible-seed runs stay reproducible on GPU (CUDA's algorithm
   auto-selection can otherwise vary run to run). -- INVESTIGATED AND
   DROPPED (August 21, 2026), not implemented. `pinn_architecture` uses
   only `nn.Linear` layers (no convolutions), so `cudnn.deterministic`
   governs conv algorithm selection this network never uses. Rather than
   implement the originally-planned fix (or the stronger
   `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG`
   alternative) on schedule regardless, tested empirically first: trained
   the same config with the same seed twice on this GPU and compared
   results directly. Outcome was bit-for-bit identical -- same `rel_l2` to
   the last digit, identical full loss history, identical model
   predictions via `torch.equal()` on a fixed grid. No evidence of GPU
   nondeterminism to fix for this architecture, so no code change was
   made; forcing `use_deterministic_algorithms(True)` would add real risk
   (can error on ops without a deterministic implementation, can slow
   training) for no measured benefit. This finding is specific to this
   machine's hardware/driver/PyTorch build (RTX 5070 Ti, driver 591.86,
   `torch==2.11.0+cu128`) -- re-verify with the same bit-for-bit
   double-run check if training ever moves to different hardware, rather
   than assuming determinism still holds.
4. Generate `requirements.txt` via `pip freeze` from `.venv`. Not a code
   change; pairs with the hardware-disclosure paragraph above. -- DONE
   (August 21, 2026), committed separately. Note for later, when
   `README.md` gets fleshed out: `torch==2.11.0+cu128` is not on plain
   PyPI -- installing this file elsewhere needs PyTorch's CUDA index URL
   passed explicitly, or `pip install -r requirements.txt` will fail to
   find that exact wheel.

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
- RESOLVED August 21, 2026: the inverse PINN diffusivity used to not be
  constrained to remain positive. Revisited the July 20, 2026 rejection of
  `log(alpha)` reparameterization after discussing the concrete trade-off
  (it changes alpha's optimization geometry -- a fixed `adam_lr` produces
  a step in alpha-space that scales with alpha's current magnitude, rather
  than the previous fixed absolute step size -- but does not invalidate
  any existing committed results, since inverse has not had a completed
  real sweep yet); decided the guarantee (alpha can no longer go negative
  or hit zero under any value the optimizer takes) was worth that
  trade-off, in preference to the other options considered (a soft
  penalty term, which needs a new hyperparameter and only discourages
  rather than guarantees; a hard clamp, which is non-differentiable and
  can hide how far an optimizer actually wanted to diverge; detect-and-
  flag only, which does not prevent wasted compute on a doomed run).
  `train_inverse` in `pinn_shared.py` now optimizes `log_alpha` (a new
  `nn.Parameter`, initialized from `torch.log(inverse_config["alpha_init"])`)
  instead of `alpha` directly; the physical diffusivity is recomputed as
  `torch.exp(log_alpha)` wherever needed. A real subtlety surfaced during
  implementation: the L-BFGS closure computes its own local `alpha =
  torch.exp(log_alpha)`, which is scoped to the closure and does NOT
  update the outer-scope `alpha` used after `optimizer_lbfgs.step(closure)`
  returns (a plain `=` inside a nested function creates its own local
  binding rather than reaching back into the enclosing scope) -- without
  an explicit `alpha = torch.exp(log_alpha)` recomputed in the outer scope
  immediately after L-BFGS finishes, every L-BFGS update to alpha would
  have been silently discarded, with the returned "alpha" permanently
  stuck at whatever it was at the end of the Adam phase. Fixed by adding
  that recompute, and by switching the post-L-BFGS diagnostic's gradient
  check from `alpha.grad` to `log_alpha.grad` (the freshly-recomputed
  `alpha` never itself participates in a `backward()` call, so it has no
  `.grad` populated -- `log_alpha` is the actual leaf Parameter that
  accumulates gradients). No API/return-shape changes: `train_inverse`
  still returns `"alpha"` as a plain float and `"history_alpha"` as a list
  of floats representing the physical (not log-space) value, so no
  notebook call sites needed to change.
  Verified via three scratch-script checks: (1) under a deliberately
  aggressive learning rate designed to destabilize a raw-alpha
  parameterization, alpha stayed strictly positive across all 300+
  recorded values even as it was driven down to ~0.004 -- confirming the
  positivity guarantee holds even in an adversarial case; (2) an isolated,
  well-conditioned synthetic probe (a trivial closure + a single L-BFGS
  step) confirmed the outer-scope recompute genuinely picks up the
  closure's optimizer update (0.1 -> 0.138), independent of whether real
  training happens to produce a large enough gradient to see the effect
  at limited float precision; (3) a normal-scale run recovered
  alpha=0.377 against true_alpha=0.4, confirming ordinary convergence
  behavior is unaffected.
- RESOLVED August 21, 2026 (see "Next Recommended Step" item 2 above for the
  full reasoning): Optuna pruning used to compare weighted training loss
  even though the loss weights vary between trials. Forward now prunes on
  a windowed-average unweighted residual; inverse now prunes on a
  windowed-average alpha error.
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
- RESOLVED August 21, 2026 (see "Next Recommended Step" item 3 above):
  baseline PINN field-evaluation time used to never be measured for the
  forward problem. Now timed per seed and reported as mean ± std, both in
  the baseline section's printout and in the final comparison table.
- New July 22, 2026, RESOLVED August 21, 2026: the forward and inverse
  Optuna search spaces used to differ in undocumented ways (inverse
  hardcoded `activation="sin"` and `N_f=10000`, and searched `N_bc`/`N_ic`
  over double forward's range). Decided and implemented -- see "Next
  Recommended Step" item 1 above for the full reasoning per dimension.
  `activation`/`N_f`/`hidden_size`/`N_bc`/`N_ic` now match forward's search
  space exactly; whether inverse specifically benefits from denser BC/IC
  coverage is deferred to a dedicated follow-up ablation rather than left
  as an unexamined asymmetry in the main comparison.
- RESOLVED August 21, 2026 (see "Next Recommended Step" item 3 above): the
  forward problem's true diffusivity (0.4) used to be a bare, repeated
  literal with no single source of truth across `pinn_shared.py`'s
  `train_forward`, the notebook's grid cell, and the FD-vs-PINN comparison
  cell. `train_forward` now reads `forward_config["true_alpha"]`, and all
  three call sites reference `forward_config["true_alpha"]`/
  `FORWARD_FIXED_CONFIG["true_alpha"]` instead of a bare `0.4`.
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
  dependency list. Now trivial to generate via `pip freeze` from the new
  `.venv`.
- New August 21, 2026: Crank-Nicolson is CPU-bound (no GPU benefit at this
  problem size) while PINN training/tuning now runs on GPU. This asymmetry is
  acceptable if disclosed (see `CLAUDE.md`'s Experimental Integrity section,
  updated the same day), but whether to also add a same-hardware CPU-only
  PINN timing as a secondary control line is still an open researcher
  decision, not yet made.

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

None pending code changes -- all GPU-migration, search-space, pruning,
timing/consolidation, and alpha-positivity fixes are committed (8 commits
ahead of `origin/methodology-cleanup` as of this write-up, not yet
pushed). `requirements.txt` was regenerated a second time today after
installing `scikit-learn` (see item 7 below) and is part of this same
uncommitted batch alongside this `PROJECT_STATUS.md` update.

## Next Recommended Step

All four GPU migration items are resolved (see "GPU Migration Action Items"
above): items 1, 2, and 4 done and verified; item 3 investigated and
deliberately dropped (empirically confirmed unnecessary for this
architecture/hardware, not implemented).

Pick up the next item on the priority-ordered backlog (ordered
biggest-impact first, per researcher preference):

1. Forward/inverse Optuna search-space asymmetry -- DECIDED AND IMPLEMENTED
   (August 21, 2026). Discussed each dimension on its merits rather than
   picking a default: `activation` and `N_f` asymmetries had no physics
   justification found (the original narrower inverse space was sized for
   MacBook CPU training time, not a deliberate scientific choice, and
   forward's own HPO run already undercuts the "sin is the right basis so
   fix it" assumption inverse was built on); `N_bc`/`N_ic` being double
   forward's range was the one dimension with a real candidate
   justification (denser boundary/IC coverage could aid alpha
   identifiability given only sparse noisy data), but that's a hypothesis,
   not a tested finding, so it wasn't kept as an unexamined asymmetry in
   the main comparison. `heat_pinn_tuned.ipynb` cell `d9980ad6` now has
   `activation` (`["tanh","sin"]`), `N_f` (`[5000,10000,20000]`),
   `hidden_size` (`[16,32,64,128]`), and `N_bc`/`N_ic` (`[100,200,400]`)
   all matching forward's search space exactly. Verified via a scratch
   Optuna study (tiny iteration counts) confirming both activations, all
   three `N_f` values, and the new `N_bc` range actually get sampled, and
   that the specific corner case that never existed in the old space
   (`hidden_size=16`, `activation="tanh"`, `N_f=5000`, `N_bc=100`) runs
   cleanly through `train_inverse`. Whether inverse specifically benefits
   from denser BC/IC coverage is deferred to a dedicated follow-up
   ablation, not resolved here. Still pending, as separate approval-gated
   steps: deciding `n_trials` for a fresh sweep (now that each trial is far
   cheaper on GPU, likely more than the current 100) and actually running
   it -- widening the search space here was free (no compute), running a
   real sweep is not.
2. Optuna pruning signal -- DECIDED AND IMPLEMENTED (August 21, 2026).
   Original problem: pruning compared weighted `total_loss` across trials,
   but `lambda_bc`/`lambda_ic`/(`lambda_data` for inverse) are themselves
   searched per-trial over a huge log-uniform range (0.001-1000), so
   raw weighted-loss magnitudes are not comparable between trials for
   reasons having nothing to do with fit quality.
   - Forward now reports the UNWEIGHTED physics/boundary/initial residual
     (`pde_loss + bc_loss + ic_loss`), averaged over the preceding 200
     iterations rather than a single instantaneous reading. Sufficient for
     forward because there is no unknown parameter to identify -- a low
     residual and a correct solution are the same thing.
   - Inverse deliberately does NOT use the same unweighted-sum approach,
     despite the appeal of symmetry: the PDE residual alone cannot
     distinguish a correctly-identified alpha from a self-consistent but
     wrong one (a flexible enough network can satisfy the PDE for the
     wrong alpha too), and `data_loss` -- the one term that actually
     disambiguates alpha -- would just be one of several equally-weighted
     terms in an unweighted sum, with no guarantee it carries enough
     influence to matter. Inverse instead reports `alpha_error`
     (`abs(alpha - true_alpha)`), windowed the same way -- the same
     quantity `objective_inverse` already uses to pick the winning trial,
     evaluated progressively instead of only at the end.
   - Considered and explicitly rejected using `alpha_error` purely because
     it uses `true_alpha`, which a real deployment would not have: this
     does not compound that concern, since pruning only decides which
     trials get cut short to save compute -- it does not determine which
     config wins (the final objective already uses `alpha_error`
     regardless of what pruning does), so pruning on the same signal
     doesn't add a new channel for ground truth to influence the result,
     it just reaches the same eventual answer faster. Whether the
     top-level inverse objective itself should move to a ground-truth-free
     criterion (e.g., held-out noisy-data fit) instead of `alpha_error` is
     a separate, bigger methodology question, logged but not decided here.
   - Windowing (200-iteration rolling average, not the instantaneous value
     at each checkpoint) applied to both, so a trial only gets flagged as
     bad if persistently worse across the whole window, not because it was
     caught at one noisy/non-monotonic instant -- alpha in particular does
     not move monotonically during training (confirmed in an earlier
     scratch test). Windowing reuses the per-iteration GPU tensors already
     being accumulated in `history_*` (from the `.item()`-removal fix
     above), so it adds no extra CPU<->GPU syncs beyond the existing one
     per 200-iteration checkpoint.
   - Verified via a scratch Optuna study (10 trials each, deliberately wide
     learning-rate range to force some trials to be clearly worse):
     pruning actually triggers for both problems (6/10 forward trials
     pruned, 4/10 inverse trials pruned), all reported intermediate values
     are finite, and completed trials' windowed values decline over
     training as expected.
3. Baseline PINN field-evaluation timing + true-diffusivity consolidation --
   DONE AND VERIFIED (August 21, 2026). `train_forward` now reads
   `forward_config["true_alpha"]` instead of a hardcoded `alpha = 0.4`
   (matching how `train_inverse` already requires `inverse_config["true_alpha"]`);
   `forward_config` and `FORWARD_FIXED_CONFIG` in `heat_pinn_tuned.ipynb`
   both carry `"true_alpha": 0.4` (mirroring how `INVERSE_OBS_CONFIG` and
   `INVERSE_FIXED_CONFIG` both already carry inverse's), and the two
   remaining bare `0.4` literals (the held-out test grid's exact solution,
   the FD-vs-PINN comparison's exact solution) now read from
   `forward_config["true_alpha"]` instead. Baseline's field-evaluation
   time is now measured per seed (previously un-timed, printing `N/A` in
   the final table) and reported as its own mean ± std cost line, both in
   the baseline section's own printout and as a new row in the final
   Baseline/Tuned/FD comparison table. Verified via a scratch script:
   confirmed a model trained with `true_alpha=0.7` actually tracks the
   alpha=0.7 exact solution (not 0.4) at a held-out point, and confirmed
   omitting `true_alpha` from a config now raises `KeyError` instead of
   silently keeping the old hardcoded value.
4. Build the planned sensitivity sweeps -- vary `noise_std` at fixed
   `N_obs`, and vary `N_obs` at fixed `noise_std` -- comparing the tuned
   PINN and CN-NLS at each point. `INVERSE_OBS_CONFIG` is already factored
   out specifically to support this.
5. Once the above are settled: actually run the real Optuna sweeps and
   retrain loops (a real, approval-gated training job) to get real baseline/
   tuned numbers, confirm they look sane, and only then remove
   `heat_pinn_basic.ipynb`.
6. Smaller polish items: `N_bc` naming clarification, fleshing out
   `README.md` (including the `torch==2.11.0+cu128` CUDA-index-URL note
   above), the unused top-level `device` variable left over in
   `heat_pinn_tuned.ipynb` cell `16dcba3f` now that model-evaluation code
   reads each model's own device instead, and the minor cosmetic-only
   naming/comment inconsistencies noted above.
7. Pre-real-sweep notebook integration check -- DONE (August 21, 2026).
   Everything above (items 1-4, plus the GPU-migration and alpha-positivity
   fixes) had only ever been verified via standalone scratch scripts
   calling `pinn_shared.py` functions directly -- `heat_pinn_tuned.ipynb`
   itself had not actually been executed top to bottom even once this
   session, so notebook-level integration bugs (stale cell references,
   execution-order issues) could not have been caught by any of that
   testing. Ran a reduced-scale dry run instead: a scratchpad copy of the
   notebook with `n_trials=3` and drastically cut iteration counts, its own
   distinctly-named Optuna storage/pickle files (so it could never collide
   with or overwrite the real ones -- confirmed none existed yet anyway),
   executed via `jupyter nbconvert --execute`. Found one real bug this way:
   `scikit-learn` was missing from `.venv`, needed internally by
   `optuna.visualization.plot_param_importances()` -- without this check,
   the real 300-trial sweep would have crashed at that exact cell *after*
   the full ~1.5-2 hour search completed, before any retrain/comparison
   cells ever ran. Installed `scikit-learn` and re-ran; the full notebook
   then executed with zero errors end to end (verified by scanning every
   cell's outputs for error-type entries -- none found), both final
   Baseline/Tuned/(FD or CN-NLS) comparison tables printed completely, and
   the FD solver and CN-NLS (unaffected by the iteration-count reduction,
   since neither depends on gradient-based training) produced properly
   accurate results -- FD rel L2 error `3.3e-07`, CN-NLS recovered
   `alpha=0.413` against `true_alpha=0.4` -- confirming those code paths
   are solid independent of this session's PINN-side changes.
   `requirements.txt` regenerated afterward to include `scikit-learn`.

## Suggested First Message to Claude

Read `CLAUDE.md` and `PROJECT_STATUS.md`, check the current Git branch and Git
status, and review the most recent commits. Explain where the project stands
and propose the next cleanup step from "Next Recommended Step" above. Do not
edit anything yet.