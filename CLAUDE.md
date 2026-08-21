# Project Instructions

## Project Purpose

This undergraduate research project compares physics-informed neural networks
(PINNs) with the Crank-Nicolson finite-difference method for the
one-dimensional heat equation.

The study includes:

1. A forward problem with known thermal diffusivity.
2. An inverse problem that estimates thermal diffusivity from sparse, noisy
   observations.
3. A literature-informed baseline PINN.
4. An Optuna-tuned PINN.
5. An efficient Crank-Nicolson forward solver.
6. A Crank-Nicolson-based inverse parameter estimator.
7. A cost-benefit analysis covering accuracy, runtime, tuning cost, training
   sensitivity, and practical complexity.

The main research question is not simply whether a PINN can solve the heat
equation. It is whether the flexibility of a PINN provides enough practical
benefit to justify its training and tuning cost when a strong classical method
already exists.

## Researcher Learning Requirement

The researcher wants to understand and explain the code personally for paper
reviewers, presentations, and technical interviews.

Do not silently rewrite large sections of the project.

For every meaningful change:

1. Inspect the current implementation.
2. Explain the problem in plain language.
3. Explain the relevant mathematical or numerical idea.
4. Propose one focused change.
5. Wait for explicit approval before editing.
6. Show the important part of the diff after editing.
7. Run a small and relevant test.
8. Explain what the test demonstrates.
9. Wait for approval before moving to the next major change.

Use clear, straightforward Python. Avoid unnecessary frameworks, abstractions,
or clever code that makes the implementation harder to explain.

## Safety and Git Rules

- The protected original version is stored on the `main` branch.
- Development work belongs on the `methodology-cleanup` branch.
- Do not switch branches without explicit approval.
- Do not commit, push, merge, open a pull request, or delete files without
  explicit approval.
- Do not run an Optuna sweep or another expensive training job without explicit
  approval.
- Do not delete or overwrite existing Optuna studies.
- Use existing best configurations for quick tests while the code is being
  cleaned up.
- Make small, focused changes rather than one large refactor.
- Preserve existing results until corrected replacements have been verified.

## Experimental Integrity

The comparison must not intentionally handicap either method.

The Crank-Nicolson baseline should use an efficient tridiagonal or banded
implementation with reusable factorization.

The baseline PINN and tuned PINN must remain separate:

- The baseline PINN uses reasonable manually selected hyperparameters based on
  standard PINN practice and literature.
- The tuned PINN uses Optuna to investigate how far PINN performance can be
  improved through additional computational effort.

Report these costs separately:

- Final PINN training time.
- PINN field-evaluation time.
- Total Optuna search time.
- Crank-Nicolson setup and solve time.
- CN-based inverse-optimization time.

Final comparisons must use reproducible seeds and identical inverse
observations for PINN and CN.

Do not select the luckiest individual PINN run as the representative result.
Report performance across repeated seeds or datasets.

Report the hardware used for each measured cost line (CPU model for CN, GPU
model for PINN). Running CN on CPU and PINN training/tuning on GPU is
acceptable -- CN does not benefit from GPU parallelism at this problem size
-- but it must be disclosed, not left implicit.

## Current Important Files

- `pinn_shared.py`: shared PINN, inverse, and Crank-Nicolson functions.
- `heat_pinn_basic.ipynb`: literature-informed baseline experiments.
- `heat_pinn_tuned.ipynb`: Optuna tuning and tuned comparisons.
- `James_Desjarlais_PINN_Final.pdf`: original submitted thesis.
- `PROJECT_STATUS.md`: current project state and handoff information.

Optuna database and pickle files may exist locally but are intentionally ignored
by Git.

## Session Startup

At the beginning of every session:

1. Read `PROJECT_STATUS.md`.
2. Check the current Git branch.
3. Check whether uncommitted changes exist.
4. Review the most recent Git commits.
5. Inspect only the files needed for the next task.
6. Explain the current state and recommended next step.
7. Do not edit anything until the researcher approves the step.

## Session Handoff

Before ending a session, update `PROJECT_STATUS.md` with:

- Date.
- Current branch.
- Work completed.
- Important decisions and why they were made.
- Files changed.
- Tests run and their results.
- Unresolved issues.
- Exact next recommended step.
- Whether uncommitted changes remain.

Keep `PROJECT_STATUS.md` concise and current. Replace outdated status
information rather than turning it into an endlessly growing transcript.

Show the proposed status update to the researcher before saving, committing, or
pushing it.