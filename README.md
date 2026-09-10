# HEAT_PINN

An independent research project comparing physics-informed neural networks
(PINNs) against the Crank-Nicolson finite-difference method on the
one-dimensional heat equation, for both:

- **Forward problem**: solving the PDE given a known thermal diffusivity.
- **Inverse problem**: estimating thermal diffusivity from sparse, noisy
  observations.

For each problem, a literature-informed baseline PINN and an Optuna-tuned
PINN are compared against an efficient Crank-Nicolson solver / CN-based
least-squares estimator, on accuracy, runtime, and tuning cost. The core
research question isn't just "can a PINN solve the heat equation," but
whether a PINN's flexibility is worth its training and tuning cost when a
strong classical method already exists.

This project began as an undergraduate thesis and has continued as
independent research since graduating -- see "Project history" below.

## Project status

The code and experiments are functionally complete: PINN/inverse/CN
implementations, Optuna hyperparameter search for both problems, leakage-safe
multi-seed evaluation, and sensitivity sweeps have all been run successfully.

**Still in progress**: the explanatory markdown and code comments in
[`heat_eqn_pinn.ipynb`](heat_eqn_pinn.ipynb) are being filled in and revised,
and the write-up (paper) has not been started yet. One more full Optuna
sweep is planned to regenerate final citable numbers now that the notebook
cleanup is complete.

## Project history

[`James_Desjarlais_PINN_Final.pdf`](James_Desjarlais_PINN_Final.pdf) is my
original undergraduate thesis, submitted as coursework. Everything else in
this repository was built afterward, as independent research -- the
methodology has been substantially reworked since (leakage-safe evaluation,
fair Optuna pruning, multi-seed reporting, disclosed hardware costs, etc.),
so the PDF should be read as the project's starting point, not its current
state. It's kept here for provenance.

## Repository contents

- [`pinn_core.py`](pinn_core.py) — PINN architecture, training loops
  (forward and inverse), sampling, losses, the Crank-Nicolson forward
  solver, and the CN-based inverse (least-squares) estimator.
- [`heat_eqn_pinn.ipynb`](heat_eqn_pinn.ipynb) — Optuna tuning,
  baseline-vs-tuned-vs-classical comparisons, and sensitivity sweeps for
  both the forward and inverse problems.
- [`James_Desjarlais_PINN_Final.pdf`](James_Desjarlais_PINN_Final.pdf) —
  the original undergraduate thesis (see "Project history" above).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.11.0+cu128` (a CUDA-enabled build), which
is not available on plain PyPI. Install PyTorch first from its own CUDA
index, then install the rest:

```bash
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

If you don't have an NVIDIA GPU, install a CPU-only `torch` build instead —
everything in this repo auto-detects GPU availability and falls back to CPU.
