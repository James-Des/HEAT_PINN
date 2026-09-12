# Project Status

**File rename note (September 10, 2026)**: `pinn_shared.py` -> `pinn_core.py`
and `heat_pinn_tuned.ipynb` -> `heat_eqn_pinn.ipynb`. Everything below this
line was written before the rename and still uses the old names -- treat
this log as a historical record, not current file paths. See `CLAUDE.md`
for the current file list.

## Last Updated

September 11, 2026

## Current Branch

`methodology-cleanup`

`main` is now a curated public snapshot kept in sync with
`methodology-cleanup` (minus `CLAUDE.md`/`PROJECT_STATUS.md`, which are
intentionally not public) -- see "Session Update (September 11, 2026)"
below for the current branching model. It is no longer "the original
project" -- that framing is stale as of the repo going public this
session.

## Session Update (September 11, 2026)

**Repo went public**: the researcher made the GitHub repo public. This
changed the branching model:
- `main` is now a curated public snapshot (fast-forwarded/merged from
  `methodology-cleanup`, then `CLAUDE.md`/`PROJECT_STATUS.md` stripped and
  gitignored on `main` only -- these internal working files are
  intentionally not public). Future syncs to `main` will hit a
  modify/delete conflict on those two files each time, which is expected;
  resolve by re-deleting them on `main`'s side.
- `methodology-cleanup` remains the full working branch, unchanged in
  spirit -- still has both files, still where all real work happens.
- `main` currently sits 4 commits behind `methodology-cleanup` (last
  synced at `a1a35b3`); not yet re-synced pending more comment cleanup.

**Renames** (`a1a35b3`): `pinn_shared.py` -> `pinn_core.py`,
`heat_pinn_tuned.ipynb` -> `heat_eqn_pinn.ipynb` -- "shared" stopped being
accurate once `heat_pinn_basic.ipynb` was retired. `README.md` rewritten:
frames the project as independent research continuing past undergrad, adds
a "Project history" section explaining the thesis PDF is the pre-cleanup
starting point (kept for provenance, not current methodology).

**Forward comparison table fixes** (`43e3c6e`, `e48cfc0`): two real gaps
found and fixed, not just cosmetic --
- Baseline never tracked per-seed L-inf error; Tuned never tracked
  per-seed inference time or L-inf across its 5 winning-config models.
  Both added, mirroring the existing per-seed rel-L2 pattern.
- Final table restyled: dropped the single-representative-run row
  (still printed separately near the plots, just not duplicated in the
  summary table). Every Baseline/Tuned cell is now a real mean +/- std;
  FD stays single-valued (deterministic, no seed dependence).
- FD convergence study (`e48cfc0`): was `fd_solver(nx, nx*2)` with no
  documented rationale, inconsistent with the final-reporting cell's
  `fd_solver(1000, 1000)`. Fixed to `fd_solver(nx, nx)`, widened
  `N_values` to `[10, 50, 100, 500, 1000, 5000]` so the sweep explicitly
  includes the actual reported resolution. Verified empirically: order-2
  convergence holds cleanly across the whole range.

**Notebook tooling issue found and partially fixed**: the notebook had
grown to ~920KB (embedded cell outputs), exceeding the 25K-token limit
Claude's `NotebookEdit` tool needs to read it before editing. This forced
raw out-of-band Python-script edits with no live sync to an open VS Code
session, which caused a real edit conflict this session -- the researcher
lost some unsaved manual comment edits when the file was reloaded/edited
externally. Acknowledged as a minor, non-recoverable loss (not committed,
not recoverable from git). Fix applied (`5248a2f`): cleared all cell
outputs, shrinking the file to ~70KB / ~27,643 tokens -- still about 10%
over the tool's limit (confirmed content-driven via a compact-JSON test,
not a formatting issue), so `NotebookEdit` is still not usable yet.
**Until the notebook drops under the token limit**, follow this protocol
when editing it together: save any pending VS Code changes before Claude
edits the file; reload/revert the file in VS Code after Claude edits it.
Re-check periodically whether ongoing comment trimming has closed the gap
-- once it does, normal live-synced editing resumes automatically.

**Comment cleanup in progress** (researcher-driven, ongoing): established
convention this session -- no em dashes in comments (or any written text)
for this researcher; use periods/commas instead. Two comment
simplifications proposed but NOT YET applied (still using old wording as
of this update): the "Move inputs to whichever device..." comment (forward
plotting cell) and the "Every Baseline/Tuned number below..." comment
(final comparison table cell) -- both have condensed replacements already
drafted in chat history, just need the researcher's go-ahead to apply.

**Outside this repo**: the researcher's GitHub profile README
(`James-Des/James-Des`) was also updated this session -- added HEAT_PINN
as the top-listed project, rewrote the bio (reflects graduation, drops
Quant Research/Trading and "seeking internships" framing, adds Data
Science/FDE/Simulation SWE/ML/Scientific Computing, and calls out
numerical PDE methods/physics-informed ML/Monte Carlo methods/deep
learning as specialties), and added SciPy/Optuna/Matplotlib to the skills
list. Not tracked in this repo's history.

## Hardware and Environment

Development has moved from a CPU-only MacBook Pro to a new Windows PC with an
NVIDIA RTX 5070 Ti (CUDA-enabled). Python 3.12 and a fresh virtual environment
(`.venv`, gitignored) were set up on this machine on August 21, 2026, with all
required packages installed, including CUDA-enabled `torch 2.11.0+cu128`
(GPU detection confirmed working). Git for Windows was also installed the same
day -- only GitHub Desktop (with its own bundled, non-PATH git) was present
before, so command-line git now works directly in a terminal/VSCode after a
restart.

**Project location changed September 9, 2026**: moved from
`C:\Users\James\OneDrive\Documents\GitHub\HEAT_PINN` to
`C:\dev\HEAT_PINN`. The OneDrive path was silently redirecting the
Windows "Documents" folder into cloud sync (a "Known Folder Move"), which
was uploading the entire 5 GB `.venv` (mostly the CUDA-bundled `torch`
install) to OneDrive on every change -- not needed, since `.venv` is
gitignored and fully reproducible from `requirements.txt`, and GitHub
already serves as the real backup for the code itself. The old OneDrive
copy was deleted after verifying the new location's git history,
uncommitted changes, and a real GPU training smoke test all matched
exactly. If GitHub Desktop can't find the repo, use its "Locate..."
button (not "Clone Again") and point it at the new path. GitHub
Desktop's own default clone location is still `Documents\GitHub`, which
would put any *new* repo back inside OneDrive -- worth changing in
GitHub Desktop's settings (Options -> General -> Local repository
storage) if that hasn't been done yet.

Prior MacBook-generated results will not be reused. All real Optuna sweeps
and final comparisons will be run fresh on this machine so every reported
cost line shares one consistent hardware baseline.

## GPU Migration Action Items

Agreed and fully completed August 21, 2026, sequenced as separate small
changes (each got its own diff and test), per the Researcher Learning
Requirement in `CLAUDE.md`.

1. Add a `device` parameter to `train_forward`/`train_inverse` in
   `pinn_shared.py`, defaulting to auto-detect (GPU if available, else CPU),
   with explicit `device="cpu"` reserved for a future control-timing run.
   DONE. The initial commit only updated the inside of these two functions;
   two call sites still assumed CPU tensors and would have crashed on this
   GPU machine -- `cn_nls_baseline`'s bare `.numpy()` calls on
   `x_obs`/`t_obs`/`u_obs` (now GPU tensors when returned from
   `train_inverse`), and three `heat_pinn_tuned.ipynb` cells calling a
   GPU-resident model on the CPU-resident shared test grid. Fixed: `cn_nls_baseline`
   now uses `.cpu().numpy()`; the three notebook cells move the test grid to
   *that specific model's* actual device (`next(model.parameters()).device`,
   not a blanket notebook-level variable, since that would break the planned
   `device="cpu"` control-timing path) and bring predictions back with
   `.cpu()` right after.
2. Remove the per-iteration `.item()` GPU sync in both training loops. DONE.
   Losses are now appended as detached GPU tensors during the loop and
   converted to floats once in bulk after training via
   `torch.stack(...).cpu().tolist()`. One subtlety: `train_inverse`'s
   `alpha` is a single `nn.Parameter` the optimizer mutates in place, so
   plain `.detach()` would make every stored history entry alias the same
   storage and collapse the whole alpha-convergence history to one repeated
   value -- fixed with `alpha.detach().clone()`. Verified via a scratch
   script confirming genuine per-iteration variation (319/320 unique values
   across 320 iterations).
3. Set `torch.backends.cudnn.deterministic = True` for GPU reproducibility.
   INVESTIGATED AND DROPPED, not implemented. `pinn_architecture` uses only
   `nn.Linear` (no convolutions), so this setting governs an op class the
   network never uses. Tested empirically instead: trained the same config/seed
   twice on this GPU and compared directly -- bit-for-bit identical `rel_l2`,
   loss history, and predictions (`torch.equal()`). No evidence of
   nondeterminism to fix, so no code change was made. Finding is specific to
   this machine's hardware/driver/PyTorch build (RTX 5070 Ti, driver 591.86,
   `torch==2.11.0+cu128`) -- re-verify with the same bit-for-bit double-run
   check if training ever moves to different hardware.
4. Generate `requirements.txt` via `pip freeze`. DONE, regenerated a second
   time after installing `scikit-learn` (found missing via a notebook dry
   run -- see "Completed Cleanup Work (August 21, 2026)" below). Note for
   `README.md`: `torch==2.11.0+cu128` is not on plain PyPI -- installing
   this file elsewhere needs PyTorch's CUDA index URL passed explicitly.

## Current Project State

The existing PINN project has been copied into a Git repository and pushed to
GitHub.

The repository currently contains:

- `pinn_shared.py`
- `heat_pinn_basic.ipynb`
- `heat_pinn_tuned.ipynb`
- `James_Desjarlais_PINN_Final.pdf`
- `README.md`
- `requirements.txt`
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
methodology until a fresh real sweep is run.

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

These issues have been identified but not yet corrected -- investigation
targets, not permission to change everything at once:

- Collocation sampling and model initialization use shared global
  random-number state. This is normal/acceptable training-procedure
  stochasticity rather than a fairness bug (July 20, 2026 discussion;
  observation generation itself was already fixed the same day).
- `N_bc` represents points per boundary, so the actual total is twice the
  configuration value -- a naming-clarity issue, not a correctness bug.
- `heat_pinn_basic.ipynb` duplicates logic that now lives in `pinn_shared.py`
  and should eventually be removed. Its blocking condition (real Optuna
  sweep run, baseline/tuned numbers confirmed sane) is now satisfied as of
  August 22, 2026 -- see "Real Sweep Results" above. Ready to remove,
  pending explicit approval.
- Minor, low-priority cosmetic-only inconsistencies (not correctness
  issues): `FORWARD_FIXED_CONFIG["lambda_pde"]` is an int (`1`) while
  `INVERSE_FIXED_CONFIG["lambda_pde"]` is a float (`1.0`), functionally
  identical; forward's mean+/-std reporting variables use a `pinn_` naming
  prefix while inverse's use `tuned_inverse_` for the analogous quantity;
  one stale comment in the representative-run plotting cell references an
  "avoid two copies" rationale that no longer quite applies.
- `README.md` is still a single placeholder sentence -- a fresh clone has
  no setup instructions. `requirements.txt` now exists (see GPU Migration
  item 4 above), including the CUDA-index-URL caveat that should be copied
  into the README when it's written.
- Whether to add a same-hardware CPU-only PINN timing as a secondary
  control line (alongside CN's already-disclosed CPU-only cost) is still an
  open researcher decision, not yet made either way.

## Completed Setup Work

- Created a private GitHub repository.
- Added a project-specific `.gitignore`.
- Preserved the original work on `main`.
- Created the `methodology-cleanup` branch.
- Added the current source files and notebooks.
- Added persistent Claude project instructions.
- Added this session-handoff file.

## Real Sweep Results (August 22, 2026) -- READ THIS FIRST NEXT SESSION

The real run (`n_trials=300` each, full retrain + all five sensitivity
sweeps) completed successfully overnight -- no errors anywhere in the
notebook, confirmed by scanning every cell's outputs. Full numbers live in
`heat_pinn_tuned.ipynb`'s own cell outputs (source of truth, not
duplicated here in full); this section is the headline summary plus what
still needs a decision.

**Forward**: Optuna search 2874.68s (~48 min), 164 completed / 131 pruned
/ 5 failed (`NaN`, not a bug -- see below). Baseline rel L2
`3.83e-3 +/- 1.81e-3`, Tuned `2.72e-4 +/- 5.43e-5` (tuning ~14x better than
baseline), FD `3.34e-7` (~800x more accurate than even the tuned PINN).
Notably, PINN's own field-evaluation time (`0.0119s`) is faster than one
full FD solve (`0.0211s`) -- FD wins decisively on accuracy, not on
inference speed. `true_alpha` sweep (`0.1/0.4/0.7/1.0`) confirms this gap
holds across the whole tested range, not just at 0.4.

**Inverse**: Optuna search 3575.42s (~60 min), 195 completed / 94 pruned /
11 failed. Baseline alpha error `5.33e-2 +/- 9.3e-4`, Tuned
`2.35e-2 +/- 1.16e-2`, CN-NLS `1.87e-2 +/- 1.22e-2` at `0.36s` per run
(~100x faster than the PINN's own 36.27s per run, before even counting the
~60 minutes of Optuna search that produced it). **CN-NLS is at least as
accurate as the tuned PINN and dramatically cheaper** -- this is a real,
legitimate research finding, not a methodology gap: three sensitivity
sweeps (noise, `N_obs`, `true_alpha`) confirm CN-NLS wins or ties across
nearly every tested condition, and the PINN's disadvantage *widens* at
higher diffusivity (`true_alpha=1.0`: PINN `8.4e-2` vs. CN-NLS `2.7e-2`).
Not perfectly one-sided though -- PINN edged out CN-NLS at a couple of
individual sweep points (e.g. `N_obs=25`), so the honest characterization
is "competitive, CN-NLS usually at least as good," not a total PINN loss.

**The leakage-prevention methodology (built July 26, 2026) caught a real
problem live, exactly as designed**: the inverse Optuna search stage's
best trial showed `alpha_error=1.23e-5` on its one fixed dataset -- the
confirmation round (5 fresh datasets) came back at `2.12e-2`, nearly 3
orders of magnitude worse, confirming that headline number was a lucky
single-dataset draw, not real generalization. The reported final number
(`2.35e-2`) is the honest one, from data never used for selection.

**`NaN` trial failures** (5 forward, 11 inverse): Optuna correctly marks
these `FAILED` (excluded from best-trial ranking, no code fix needed) --
real hyperparameter combinations that diverged during training, mostly
clustered around extreme `lambda_bc`/`lambda_ic` values or (for inverse)
small `hidden_size` with aggressive `adam_lr`. Worth a mention in the
paper's practical-complexity discussion: PINN training can diverge
outright for unlucky hyperparameter draws; CN never does.

**Validated an earlier decision**: all top-5 inverse trials independently
chose `activation="tanh"`, confirming the August 21 decision to widen
inverse's search space to include it (previously hardcoded to `"sin"`).

**Superseded note (added September 9, 2026)**: the numbers above are from
a real completed run and are directionally trustworthy, but they are
*not* the numbers that should be quoted in the paper. Since this run,
`heat_pinn_tuned.ipynb` has picked up cosmetic fixes, two new CPU-control
timing cells, and a log-scale fix to the Optuna history plots (see
"Session Update (September 9, 2026)" below) -- one more full 300-trial
run is still needed to produce the actual final, citable numbers, now
that the notebook reflects the cleaned-up code.

### Things that need to be addressed next session

1. ~~The executed notebook is not yet committed.~~ Done -- committed
   August 22/23 (`0a05409`), and the cleanup work since then (see below)
   is committed and pushed as of September 9, 2026.
2. **Decide the paper's framing for the inverse result.** CN-NLS matching
   or beating the tuned PINN, ~100x cheaper, is a legitimate finding but
   changes what the paper's conclusion should say for the inverse problem
   specifically -- still an open decision, worth making once the final
   run's numbers are in (not something to quietly smooth over) before
   writing the discussion/conclusion section.
3. ~~Retire `heat_pinn_basic.ipynb`?~~ Done -- removed September 9, 2026,
   recoverable from git history if ever needed.
4. **CPU-control-timing run** -- decided (add it) and implemented
   September 9, 2026. Two new cells, one per problem, reuse the exact
   winning frozen config/seeds/data from the GPU final results with
   `device=torch.device("cpu")` explicitly, so the same-hardware
   comparison point CLAUDE.md's Experimental Integrity section calls for
   is now in the notebook. Currently unexecuted -- will run for real
   during the next full sweep.
5. **Smaller polish** -- mostly done September 9, 2026: `N_bc`/`N_ic`
   per-boundary clarity (see note below on a correction to this),
   `lambda_pde` int/float consistency, and the dead `device` variable
   (renamed to `default_device`) are all fixed. `README.md` is still a
   placeholder -- still open.
6. Consider whether the `NaN`-failure hyperparameter patterns are worth a
   deliberate closer look (which specific combinations diverge and why)
   as a small piece of the practical-complexity write-up, or just a
   passing mention -- not yet decided either way.

## Completed Cleanup Work (August 21, 2026)

The largest single session so far -- migrating to new hardware, then working
through the full priority-ordered methodology backlog. All changes are in
`pinn_shared.py` and `heat_pinn_tuned.ipynb` unless noted.

- Migrated development to a new Windows PC (Ryzen 9950X, RTX 5070 Ti). Set
  up `.venv`, verified CUDA-enabled torch, installed Git for Windows
  (previously only GitHub Desktop's bundled, non-PATH git existed) and
  `scikit-learn` (found missing via a notebook dry run, needed by
  `optuna.visualization.plot_param_importances()`). `requirements.txt`
  generated and later regenerated to include `scikit-learn`.
- Completed the full GPU migration -- see "GPU Migration Action Items"
  above for the four-item detail (device param + device-mismatch fixes,
  `.item()`-sync removal, the `cudnn.deterministic` investigation-and-drop,
  `requirements.txt`).
- Aligned inverse's Optuna search space with forward's exactly
  (`activation`, `N_f`, `hidden_size`, `N_bc`/`N_ic` all matched) after
  finding no physics justification for the prior narrower space -- it was
  sized for MacBook CPU time, not a deliberate scientific choice, and
  forward's own completed sweep already showed `tanh` beating `sin`, which
  undercut the assumption inverse's fixed `sin` was built on. Whether
  inverse specifically benefits from denser BC/IC coverage (the one
  dimension with a real candidate physics justification) is deferred to a
  future ablation rather than kept as an unexamined asymmetry.
- Bumped `n_trials` 100 -> 300 for both problems, grounded in a real timing
  benchmark on this GPU (~10-20s per full-sized forward trial, ~13-20s
  inverse) showing 300 trials lands around 1.5-2 hours combined with
  pruning -- versus roughly 6 hours total for 100 trials on the old
  MacBook's CPU. Pruning parameters (`n_startup_trials`, `n_warmup_steps`)
  left unchanged, since they answer questions about training dynamics that
  don't change with total trial count.
- Fixed Optuna pruning to compare trials fairly. Previously compared raw
  weighted `total_loss` across trials, but the loss weights are themselves
  searched per-trial over a huge log-uniform range, so magnitudes weren't
  comparable for reasons unrelated to fit quality. Forward now prunes on a
  200-iteration windowed average of the *unweighted* physics/boundary/
  initial residual (sufficient since forward has no unknown parameter to
  identify). Inverse prunes on a windowed average of `alpha_error` instead
  -- the unweighted residual alone can't distinguish a correctly-identified
  alpha from a self-consistent but wrong one, since only `data_loss`
  actually disambiguates alpha and would be diluted in an unweighted sum.
  `alpha_error` is the same quantity `objective_inverse`'s final selection
  already uses, evaluated progressively rather than only at the end --
  confirmed this doesn't introduce a new ground-truth-dependence channel,
  since pruning only affects which trials get cut short, not which config
  wins. Windowing (not instantaneous per-checkpoint values) applied to both,
  since alpha doesn't move monotonically during training. Verified via a
  scratch Optuna study forcing some trials to be clearly worse: pruning
  triggered for both problems (6/10 forward, 4/10 inverse), all reported
  values finite and declining as expected.
- Consolidated forward's hardcoded `true_alpha=0.4` literal into
  `forward_config`/`FORWARD_FIXED_CONFIG` (matching how inverse's
  `INVERSE_OBS_CONFIG`/`INVERSE_FIXED_CONFIG` already do), and `train_forward`
  now reads `forward_config["true_alpha"]` instead of a hardcoded local
  `alpha = 0.4`. Added baseline PINN field-evaluation timing for forward
  (measured per seed, previously un-timed and printed `N/A`). Verified a
  model trained with `true_alpha=0.7` genuinely tracks the 0.7 exact
  solution (not 0.4), and that omitting `true_alpha` now raises `KeyError`.
- Reparameterized inverse's `alpha` as `log_alpha` (`alpha = exp(log_alpha)`
  computed fresh wherever needed) so the recovered diffusivity can never go
  negative or hit zero -- revisited the July 20, 2026 rejection of this
  approach after discussing the concrete trade-off (changes alpha's
  optimization geometry; a fixed `adam_lr` now produces a step in
  alpha-space scaling with alpha's current magnitude) versus the
  alternatives (soft penalty: needs a new hyperparameter, only discourages
  rather than guarantees; hard clamp: non-differentiable, can hide how far
  an optimizer wanted to diverge; detect-and-flag only: doesn't prevent
  wasted compute). A real bug surfaced during implementation: the L-BFGS
  closure computes its own local `alpha = torch.exp(log_alpha)`, which is
  scoped to the closure and does not update the outer-scope `alpha` used
  after `optimizer_lbfgs.step(closure)` returns -- every L-BFGS update
  would have been silently discarded without an explicit recompute added in
  the outer scope immediately after L-BFGS finishes. Fixed, plus switched
  the post-L-BFGS diagnostic's gradient check from `alpha.grad` to
  `log_alpha.grad` (the recomputed `alpha` never itself participates in a
  `backward()` call). No API changes -- `train_inverse` still returns
  `"alpha"` as a plain float. Verified via three checks: alpha stayed
  strictly positive under a deliberately destabilizing learning rate even
  as it was driven to ~0.004; an isolated synthetic probe confirmed the
  outer-scope recompute genuinely picks up the closure's update; a
  normal-scale run recovered `alpha=0.377` against `true_alpha=0.4`.
- Added five sensitivity analyses to `heat_pinn_tuned.ipynb`, all freezing
  the already-tuned winning config rather than re-running HPO per point
  (testing robustness of the selected config, not repeating the expensive
  search), and all reusing already-computed default-point results instead
  of retraining duplicates of the identical experiment:
  - Inverse noise sweep (cell `157f354e`): `noise_std` in
    `[0.01, 0.05, 0.1, 0.2]` at fixed `N_obs=50`.
  - Inverse observation-count sweep (cell `1b6c3646`): `N_obs` in
    `[10, 25, 50, 100]` at fixed `noise_std=0.05`.
  - Forward `true_alpha` sweep (cell `5f58a1ce`): `true_alpha` in
    `[0.1, 0.4, 0.7, 1.0]` against FD (needs no retraining, `fd_solver`
    takes `alpha` directly) -- forward previously had zero sensitivity
    analysis of its own.
  - Inverse `true_alpha` sweep (cell `69c40d92`): same four values, with
    `alpha_init` deliberately kept fixed at its already-tuned value rather
    than scaling with the swept alpha, since a real deployment of this
    tuned recipe would not know to adjust `alpha_init` either -- the
    realistic generalization test, not an unfair handicap. All four values
    stay within CN-NLS's fixed `[0.01, 1.0]` search range.
  - Seed ranges kept disjoint across every stage and sweep: search=0,
    confirmation=1-5, final-eval=6-10, noise sweep=11-15, N_obs sweep=16-20,
    forward alpha sweep=20-24, inverse alpha sweep=21-25.
  - Confirmed in passing: the "test top-3 candidates across 5 seeds, pick
    winner by mean performance" method was already implemented for both
    problems before this session's sweep work began -- kept at top-3
    rather than bumped to top-5, per researcher choice.
- Verified the complete notebook end-to-end three times via reduced-scale
  `jupyter nbconvert --execute` dry runs (n_trials=3, cut iteration counts,
  isolated scratch-only Optuna storage/pickle files that could never
  collide with the real ones) as each round of cells was added. Final pass:
  zero errors across the whole notebook, both final comparison tables
  printed completely, and FD/CN-NLS (unaffected by the dry run's PINN-side
  iteration-count reduction) produced properly accurate results at every
  check -- e.g. FD rel L2 error `3.3e-07`, CN-NLS recovering
  `alpha=0.413` against `true_alpha=0.4`. The `scikit-learn` gap (see
  above) was caught by the first of these dry runs; without it, the real
  300-trial sweep would have crashed after the full search completed,
  before any retrain/comparison cells ran.
  - Tooling note for future sessions: the notebook has grown too large for
    the Read tool to process in one call, which `NotebookEdit` requires.
    The third dry run's scratch copy was prepared by directly rewriting
    cell sources via a Python/json script instead, applied only to a
    throwaway file outside the repo -- not a change to how the real
    notebook gets edited, but a workaround worth remembering if more cells
    get added later.
- Confirmed the real notebook itself was never touched by any dry-run
  reduction (all three dry runs only ever modified throwaway scratchpad
  copies) -- `n_trials=300`, real iteration counts, and no `DRYRUN`
  references anywhere, verified directly against the file before ending
  the session.

Testing throughout was standalone scratch scripts (not part of the repo)
for `pinn_shared.py`-level changes, plus three full-notebook
`jupyter nbconvert --execute` dry runs at reduced scale for notebook-level
integration, as described above. No real Optuna sweep has been run yet.

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
  ahead of planned follow-up sensitivity sweeps (built out August 21, 2026
  -- see above).
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

## Session Update (September 9, 2026)

Two commits this session, both pushed to `origin/methodology-cleanup`.

**Commit 1 (`dd7c2a6`)** -- pre-final-run cleanup, done before touching
anything expensive:
- Added two CPU-only control-timing cells (one per problem), reusing the
  exact winning frozen config/seeds/data from each problem's GPU final
  results with `device=torch.device("cpu")` explicitly.
- Retired `heat_pinn_basic.ipynb` (fully superseded by `pinn_shared.py`).
- Fixed `lambda_pde` int/float inconsistency between `FORWARD_FIXED_CONFIG`
  and `INVERSE_FIXED_CONFIG`, and renamed the unused top-level `device`
  variable to `default_device` (it was never threaded through to control
  training -- each function auto-detects or takes an explicit override).
- Added an `N_bc`/`N_ic` per-boundary clarity docstring/comment.
  **Correction caught by the researcher**: the first version of this
  comment incorrectly implied `N_ic` is also doubled like `N_bc`. Only
  `N_bc` is (`sample_points()` samples it once each for `x=0` and `x=1`);
  `N_ic` is a single set of points at `t=0` with no doubling. Fixed in
  both `pinn_shared.py` and the notebook.

**Commit 2 (pending, this session)** -- notebook documentation pass:
- Fixed the Optuna `plot_optimization_history` plots (both problems) to
  use a log-scale y-axis -- objective values span orders of magnitude, so
  a linear axis crowded nearly every trial against the best-value line.
  `plot_optimization_history` has no built-in log-scale option, so the
  returned plotly figure is updated directly (`fig.update_yaxes(type="log")`).
- Added 16 `**TODO:**` markdown placeholder cells throughout the notebook,
  each a one-sentence prompt for the researcher to fill in with their own
  explanation (methodology "why" cells now; result-interpretation cells
  marked to wait until after the final run produces real numbers).
  Deliberately not added everywhere -- skipped cells that already had
  thorough inline code comments, to avoid padding.
  Established workflow going forward: researcher writes a rough first
  pass, Claude checks grammar/spelling and, more importantly, checks
  factual claims against the actual code and flags anything inaccurate
  rather than just polishing over it.
- Notebook-level overview cell (the first TODO) filled in collaboratively
  -- written, then revised twice for tone (cut em dashes, "pipeline",
  italics, and AI-sounding rhetorical framing per researcher feedback) and
  for a stronger opening line the researcher drafted and Claude polished.
- Researcher then hand-edited comments throughout several cells directly
  in VS Code (trimming several of Claude's longer rationale-comments down
  to shorter, plainer versions, and adding new ones -- e.g. an Optuna
  search-space explanation, a median-pruner explanation). Reviewed for
  accuracy; all correct except the `N_bc`/`N_ic` case above.

**Also this session**: the whole project was relocated from OneDrive to
`C:\dev\HEAT_PINN` -- see "Hardware and Environment" above for why and
how. Verified before deleting the old copy: git history, uncommitted
changes, and a real GPU training smoke test all matched exactly at the
new location.

**Known small inconsistency, not yet fixed**: the log-scale plot fix's
explanatory comment ended up only in the inverse cell (`7b5e8b69`) after
the researcher's hand-editing pass -- the forward cell (`e8d3682d`) lost
its version of that comment. Not a functional issue (the log-scale code
itself is identical and correct in both), just a documentation asymmetry
worth a one-line fix whenever convenient.

## Current Uncommitted Changes

None as of this update -- everything through the September 11, 2026
session is committed and pushed to `origin/methodology-cleanup` (up to
`5248a2f`). `main` is 4 commits behind, not yet re-synced.

## Next Recommended Step

Continue the researcher-driven comment-cleanup pass on `heat_eqn_pinn.ipynb`
(in progress) -- apply the two pending comment simplifications noted in
"Session Update (September 11, 2026)" above if still wanted, and keep the
save-before-edit/reload-after-edit protocol until the notebook drops under
NotebookEdit's token limit. Several `**TODO:**` markdown cells are also
still unfilled (grep for `TODO`). Once the researcher is satisfied with
that pass (or decides to finish it after the run instead -- either order
works since these edits never require a rerun), the next real blocker is
the **final full run**: 300 trials for both problems, all sensitivity
sweeps, including the two CPU-control cells, the log-scale plot fix, and
this session's table/convergence-study fixes. This is the expensive step
(roughly 1.5-2 hours based on the August 22 run) and needs explicit
researcher approval to kick off, per `CLAUDE.md`. After that: commit the
executed notebook, re-sync `main`, decide the inverse result's paper
framing (still open, see item 2 further up), and then the paper itself.

## Suggested First Message to Claude

Read `CLAUDE.md` and `PROJECT_STATUS.md`, check the current Git branch and Git
status, and review the most recent commits. Explain where the project stands
and propose the next cleanup step from "Next Recommended Step" above. Do not
edit anything yet.
