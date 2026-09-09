"""
pinn_shared.py

Shared logic for the heat-equation PINN vs. Crank-Nicolson study.
Import this from both heat_pinn_basic.ipynb (untuned baselines) and
heat_pinn_tuned.ipynb (HPO-tuned runs + CN-NLS baseline), so both
notebooks run the literal same implementation, differing only in the
config dict passed in.

Every function below is a verbatim copy of what was already in
heat_pinn_tuned.ipynb, with ONE exception: set_seed() is new (neither
notebook had a reusable seeding function; heat_pinn_basic.ipynb had no
seeding at all). Everywhere else, changes are marked "# NEW" inline and
are additive only -- nothing was rewritten, reformatted, or restructured.

Changes, and why each was necessary:
  1. set_seed(seed) -- new function. Call before any sampling/training.
  2. train_inverse() -- added a local `import optuna` inside the
     trial-pruning branch. This isn't a style choice: once this function
     moved out of the notebook, `optuna.exceptions.TrialPruned()` needs
     `optuna` bound in *this* module's namespace, not the notebook's --
     Python resolves free variables against the module a function was
     defined in. Also added an early-stop diagnostic after the L-BFGS
     phase (flagging heat_pinn_basic's observed 3-call termination), and
     added `alpha_error` / `lbfgs_calls` to the return dict (additive,
     nothing removed).
  3. cn_nls_baseline() -- added wall-clock timing (the original had
     none). Return signature is UNCHANGED (still a 3-tuple) so existing
     call sites that unpack `best_alpha, mse_history, alpha_candidates =
     cn_nls_baseline(...)` do not need to change.
  4. train_forward() -- moved here from heat_pinn_tuned.ipynb (forward
     counterpart to train_inverse). Same local `import optuna` fix as
     train_inverse, and for the same reason.

fd_solver(), pinn_architecture, sample_points, generate_noisy_data,
compute_loss, and compute_loss_inverse are byte-for-byte what was already
in heat_pinn_tuned.ipynb -- no reformatting, no renamed variables.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags
from scipy.sparse.linalg import splu


def set_seed(seed: int) -> None:
    """NEW. Seed torch and numpy. Call at the start of every experiment
    you intend to report a number from."""
    torch.manual_seed(seed)
    np.random.seed(seed)

class pinn_architecture(nn.Module):
    def __init__(self, hidden_size=20, n_layers=4, activation='tanh'):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(n_layers - 1)]
        )
        self.output_layer = nn.Linear(hidden_size, 1)

        if activation == 'tanh':
            self.act = torch.tanh
        elif activation == 'sin':
            self.act = torch.sin
        elif activation == 'swish':
            self.act = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        out = self.act(self.input_layer(xt))
        for layer in self.hidden_layers:
            out = self.act(layer(out))
        return self.output_layer(out)

def sample_points(N_f, N_bc, N_ic):
    # N_bc is points PER boundary (x=0 and x=1 each get N_bc), so the
    # actual total boundary-point count sampled below is 2 * N_bc.
    x_f = torch.rand(N_f ,1)
    t_f = torch.rand(N_f, 1)
    
    x_bc_0 = torch.zeros(N_bc, 1)
    x_bc_1 = torch.ones(N_bc, 1)
    x_bc = torch.cat([x_bc_0, x_bc_1], dim = 0)
    
    t_bc = torch.cat([torch.rand(N_bc, 1), torch.rand(N_bc, 1)], dim=0)
    
    x_ic = torch.rand(N_ic, 1)
    t_ic = torch.zeros(N_ic, 1)
    
    return x_f, t_f, x_bc, t_bc, x_ic, t_ic


def generate_noisy_data(inverse_config):
    x_obs = torch.rand(inverse_config["N_obs"], 1)
    t_obs = torch.rand(inverse_config["N_obs"], 1)
    
    u_exact = torch.sin(torch.pi * x_obs) * torch.exp(-inverse_config["true_alpha"] * torch.pi**2 * t_obs)
    
    noise = torch.randn(inverse_config["N_obs"], 1) * inverse_config["noise_std"]
    
    u_obs = u_exact + noise
    
    return x_obs, t_obs, u_obs

def compute_loss(model, alpha, x_f, t_f, x_bc, t_bc, x_ic, t_ic,
                 lambda_pde=1.0, lambda_bc=1.0, lambda_ic=1.0):
    x_f.requires_grad_(True)
    t_f.requires_grad_(True)
    
    #get u values from model
    u_f = model(x_f,t_f)
    
    #compute x derivatives using autograd
    u_x = torch.autograd.grad(u_f, x_f, grad_outputs = torch.ones_like(u_f), create_graph = True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs = torch.ones_like(u_x), create_graph = True)[0]
    
    #compute t derivative the same way
    u_t = torch.autograd.grad(u_f, t_f, grad_outputs = torch.ones_like(u_f), create_graph = True)[0]
    
    #pde loss from pde residual
    pde_residual = u_t - alpha * u_xx
    pde_loss = torch.mean(pde_residual**2)
    
    #boundary condition loss 
    u_bc = model(x_bc, t_bc)
    bc_loss = torch.mean(u_bc**2)
    
    #initial conditions loss from start u(x,0) = sin(pi*x)
    u_ic = model(x_ic, t_ic)
    true_ic = torch.sin(torch.pi * x_ic)
    ic_loss = torch.mean((u_ic - true_ic)**2)
    
    total_loss = lambda_pde * pde_loss + lambda_bc * bc_loss + lambda_ic * ic_loss
    
    return total_loss, pde_loss, bc_loss, ic_loss

def train_forward(forward_config, print_training=True, trial=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    activation = forward_config.get("activation", "tanh")
    lambda_pde = forward_config.get("lambda_pde", 1.0)
    lambda_bc  = forward_config.get("lambda_bc",  1.0)
    lambda_ic  = forward_config.get("lambda_ic",  1.0)

    model = pinn_architecture(forward_config["hidden_size"], forward_config["n_layers"], activation).to(device)

    alpha = forward_config["true_alpha"]

    optimizer = torch.optim.Adam(model.parameters(), lr = forward_config["adam_lr"])

    x_f, t_f, x_bc, t_bc, x_ic, t_ic = sample_points(forward_config["N_f"], forward_config["N_bc"], forward_config["N_ic"])
    x_f, t_f, x_bc, t_bc, x_ic, t_ic = (
        x_f.to(device), t_f.to(device), x_bc.to(device), t_bc.to(device), x_ic.to(device), t_ic.to(device)
    )

    N_iters = forward_config["adam_iters"]

    history_total = []
    history_pde = []
    history_bc = []
    history_ic = []

    pinn_train_start = time.time()

    for i in range(1, N_iters + 1):
        optimizer.zero_grad()
        
        total_loss, pde_loss, bc_loss, ic_loss = compute_loss(
            model, alpha, x_f, t_f, x_bc, t_bc, x_ic, t_ic,
            lambda_pde, lambda_bc, lambda_ic)
        
        total_loss.backward()
        optimizer.step()

        # Store detached GPU tensors during the loop instead of calling
        # .item() every iteration -- .item() forces a CPU<->GPU sync that
        # stalls the GPU pipeline. Each of these is a fresh tensor from
        # this iteration's forward pass (not a Parameter mutated in place),
        # so plain .detach() is safe -- converted to floats once, in bulk,
        # after training finishes.
        history_total.append(total_loss.detach())
        history_pde.append(pde_loss.detach())
        history_bc.append(bc_loss.detach())
        history_ic.append(ic_loss.detach())

        if print_training and i % 200 == 0:
            print(f"Iter {i} | Total: {total_loss.item():.4e} | PDE: {pde_loss.item():.4e} | BC: {bc_loss.item():.4e} | IC: {ic_loss.item():.4e}")

        if trial is not None and i % 200 == 0:
            import optuna  # local import: only needed on this path, and must be
                            # bound in this module now that train_forward no longer
                            # lives in the same notebook namespace as `import optuna`
            # Report the UNWEIGHTED physics/boundary/initial residual,
            # averaged over the preceding 200 iterations, instead of the
            # single-instant weighted total_loss. Unweighted because
            # lambda_bc/lambda_ic are searched per-trial over a huge
            # log-uniform range, so comparing raw weighted total_loss
            # across trials compares numbers that differ for reasons
            # having nothing to do with fit quality. Windowed (not just
            # the instantaneous value at this step) because a single
            # noisy reading can unfairly prune a trial that's mid-
            # fluctuation rather than genuinely behind -- averaging over
            # the last 200 iterations only flags trials that are
            # persistently worse, not momentarily unlucky.
            window_pde = torch.stack(history_pde[-200:])
            window_bc = torch.stack(history_bc[-200:])
            window_ic = torch.stack(history_ic[-200:])
            unweighted_avg = (window_pde + window_bc + window_ic).mean().item()
            trial.report(unweighted_avg, i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    adam_iters = len(history_total)

    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=forward_config["lbfgs_iters"])
    lbfgs_iter = [0]

    def closure():
        optimizer_lbfgs.zero_grad()
        total_loss, pde_loss, bc_loss, ic_loss = compute_loss(
            model, alpha, x_f, t_f, x_bc, t_bc, x_ic, t_ic,
            lambda_pde, lambda_bc, lambda_ic)
        total_loss.backward()

        history_total.append(total_loss.detach())
        history_pde.append(pde_loss.detach())
        history_bc.append(bc_loss.detach())
        history_ic.append(ic_loss.detach())

        lbfgs_iter[0] += 1

        if print_training and lbfgs_iter[0] % 10 == 0:
            print(f"L-BFGS Iter {lbfgs_iter[0]} | Total: {total_loss.item():.4e} | PDE: {pde_loss.item():.4e} | BC: {bc_loss.item():.4e} | IC: {ic_loss.item():.4e}")

        return total_loss

    optimizer_lbfgs.step(closure)

    pinn_train_time = time.time() - pinn_train_start

    # Convert the accumulated per-iteration loss tensors to plain floats
    # once, in bulk, now that training is done -- a single CPU<->GPU sync
    # instead of one every iteration.
    history_total = torch.stack(history_total).cpu().tolist()
    history_pde = torch.stack(history_pde).cpu().tolist()
    history_bc = torch.stack(history_bc).cpu().tolist()
    history_ic = torch.stack(history_ic).cpu().tolist()

    # Validation grid for HPO/model-selection (Optuna objective + the
    # top-configs retrain loop below) -- offset by half a grid cell from
    # the final comparison cell's grid (torch.linspace(0, 1, 1000)), so no
    # point used to pick a winning model is ever reused as a "final" test
    # point. Without this offset, the same 1000x1000 points would both
    # choose the winning hyperparameters and report how accurate the
    # winner is, which biases the reported accuracy optimistically.
    x_eval = torch.linspace(0.0005, 0.9995, 1000, device=device)
    t_eval = torch.linspace(0.0005, 0.9995, 1000, device=device)
    X_eval, T_eval = torch.meshgrid(x_eval, t_eval, indexing='ij')
    x_flat = X_eval.reshape(-1, 1)
    t_flat = T_eval.reshape(-1, 1)

    with torch.no_grad():
        u_pred = model(x_flat, t_flat)
    u_exact = torch.sin(torch.pi * x_flat) * torch.exp(-alpha * (torch.pi**2) * t_flat)
    rel_l2 = (torch.norm(u_pred - u_exact) / torch.norm(u_exact)).item()

    
    print(f"\nPINN total training time: {pinn_train_time:.2f}s")
    #print(f"Adam iterations: {adam_iters} | L-BFGS closure calls: {lbfgs_iter[0]}")
    print(f"Rel L2 error: {rel_l2:.4e}")

    return {
        "model": model,
        "history_total": history_total,
        "history_pde": history_pde,
        "history_bc": history_bc,
        "history_ic": history_ic,
        "adam_iters": adam_iters,
        "train_time": pinn_train_time,
        "rel_l2": rel_l2,
    }

def compute_loss_inverse(model, alpha, x_f, t_f, x_bc, t_bc, x_ic, t_ic, x_obs, t_obs, u_obs,
                         lambda_pde=1.0, lambda_bc=1.0, lambda_ic=1.0, lambda_data=1.0):
    x_f.requires_grad_(True)
    t_f.requires_grad_(True)
    
    u_f = model(x_f, t_f)
    
    u_x = torch.autograd.grad(u_f, x_f, grad_outputs=torch.ones_like(u_f), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_t = torch.autograd.grad(u_f, t_f, grad_outputs=torch.ones_like(u_f), create_graph=True)[0]
    
    pde_residual = u_t - alpha * u_xx
    pde_loss = torch.mean(pde_residual**2)
    
    u_bc = model(x_bc, t_bc)
    bc_loss = torch.mean(u_bc**2)
    
    u_ic = model(x_ic, t_ic)
    true_ic = torch.sin(torch.pi * x_ic)
    ic_loss = torch.mean((u_ic - true_ic)**2)
    
    u_pred_obs = model(x_obs, t_obs)
    data_loss = torch.mean((u_pred_obs - u_obs)**2)
    
    total_loss = lambda_pde * pde_loss + lambda_bc * bc_loss + lambda_ic * ic_loss + lambda_data * data_loss
    
    return total_loss, pde_loss, bc_loss, ic_loss, data_loss

def train_inverse(inverse_config, x_obs, t_obs, u_obs, print_training=True, trial=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    activation  = inverse_config.get("activation",   "tanh")
    lambda_pde  = inverse_config.get("lambda_pde",   1.0)
    lambda_bc   = inverse_config.get("lambda_bc",    1.0)
    lambda_ic   = inverse_config.get("lambda_ic",    1.0)
    lambda_data = inverse_config.get("lambda_data",  1.0)

    model = pinn_architecture(inverse_config["hidden_size"], inverse_config["n_layers"], activation).to(device)
    # log_alpha, not alpha, is the actual learnable parameter -- this
    # guarantees the physical diffusivity (exp(log_alpha), computed fresh
    # wherever alpha is needed below) can never go negative or hit zero,
    # for any value log_alpha takes during optimization. inverse_config
    # ["alpha_init"] is still the physical initial guess (must stay > 0);
    # only its log is what actually gets optimized.
    log_alpha = nn.Parameter(torch.log(torch.tensor(inverse_config["alpha_init"], device=device)))
    optimizer = torch.optim.Adam(list(model.parameters()) + [log_alpha], lr=inverse_config["adam_lr"])

    x_f, t_f, x_bc, t_bc, x_ic, t_ic = sample_points(inverse_config["N_f"], inverse_config["N_bc"], inverse_config["N_ic"])
    x_f, t_f, x_bc, t_bc, x_ic, t_ic = (
        x_f.to(device), t_f.to(device), x_bc.to(device), t_bc.to(device), x_ic.to(device), t_ic.to(device)
    )
    x_obs, t_obs, u_obs = x_obs.to(device), t_obs.to(device), u_obs.to(device)

    history_total = []
    history_pde = []
    history_bc = []
    history_ic = []
    history_data = []
    history_alpha = []

    inverse_train_start = time.time()

    for i in range(1, inverse_config["adam_iters"] + 1):
        optimizer.zero_grad()
        # Recomputed fresh every iteration from the current log_alpha, so
        # autograd traces through exp() correctly and the physical value
        # used in the physics loss always reflects the latest update.
        alpha = torch.exp(log_alpha)
        total_loss, pde_loss, bc_loss, ic_loss, data_loss = compute_loss_inverse(
            model, alpha, x_f, t_f, x_bc, t_bc, x_ic, t_ic, x_obs, t_obs, u_obs,
            lambda_pde, lambda_bc, lambda_ic, lambda_data)
        total_loss.backward()
        optimizer.step()

        # Store detached GPU tensors during the loop instead of calling
        # .item() every iteration -- .item() forces a CPU<->GPU sync that
        # stalls the GPU pipeline. The loss tensors are fresh objects from
        # this iteration's forward pass, so plain .detach() is safe. alpha
        # is now also a fresh tensor each iteration (torch.exp allocates
        # new storage, it does not return a view into log_alpha), so it no
        # longer aliases anything the optimizer mutates in place the way
        # the raw Parameter used to -- .clone() is technically no longer
        # required here, but kept anyway for defensive consistency with
        # the same pattern used everywhere else in this loop. Converted to
        # floats once, in bulk, after training finishes.
        history_total.append(total_loss.detach())
        history_pde.append(pde_loss.detach())
        history_bc.append(bc_loss.detach())
        history_ic.append(ic_loss.detach())
        history_data.append(data_loss.detach())
        history_alpha.append(alpha.detach().clone())

        if print_training and i % 200 == 0:
            print(f"Iter {i} | Total: {total_loss.item():.4e} | PDE: {pde_loss.item():.4e} | BC: {bc_loss.item():.4e} | IC: {ic_loss.item():.4e} | Alpha: {alpha.item():.4f}")

        if trial is not None and i % 200 == 0:
            import optuna  # local import: only needed on this path, and must be
                            # bound in this module now that train_inverse no longer
                            # lives in the same notebook namespace as `import optuna`
            # Report alpha error (the SAME quantity objective_inverse
            # ultimately selects the winning trial on), averaged over the
            # preceding 200 iterations, instead of weighted total_loss.
            # Not the unweighted physics/boundary/initial/data residual
            # either: the PDE residual alone cannot distinguish a
            # correctly-identified alpha from a self-consistent but wrong
            # one (a flexible enough network can satisfy the PDE for the
            # wrong alpha too), and data_loss -- the one term that actually
            # disambiguates alpha -- would just be one of several equally-
            # weighted terms in an unweighted sum, with no guarantee it
            # carries enough influence to matter. alpha_error sidesteps
            # that identifiability gap by measuring the thing we actually
            # care about directly. Windowed for the same reason as
            # forward: a single instant can be unlucky (alpha does not
            # move monotonically), so average over the last 200 iterations
            # to only flag trials that are persistently off.
            window_alpha = torch.stack(history_alpha[-200:])
            alpha_error_avg = (window_alpha - inverse_config["true_alpha"]).abs().mean().item()
            trial.report(alpha_error_avg, i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    adam_iters = len(history_total)

    optimizer_lbfgs = torch.optim.LBFGS(list(model.parameters()) + [log_alpha], lr=1.0, max_iter=inverse_config["lbfgs_iters"])
    lbfgs_iter = [0]

    def closure():
        optimizer_lbfgs.zero_grad()
        # Local to this closure -- does NOT update the outer-scope `alpha`
        # below, since a plain `=` inside a nested function creates its own
        # local binding rather than reaching back into the enclosing scope.
        alpha = torch.exp(log_alpha)
        total_loss, pde_loss, bc_loss, ic_loss, data_loss = compute_loss_inverse(
            model, alpha, x_f, t_f, x_bc, t_bc, x_ic, t_ic, x_obs, t_obs, u_obs,
            lambda_pde, lambda_bc, lambda_ic, lambda_data)
        total_loss.backward()

        history_total.append(total_loss.detach())
        history_pde.append(pde_loss.detach())
        history_bc.append(bc_loss.detach())
        history_ic.append(ic_loss.detach())
        history_data.append(data_loss.detach())
        history_alpha.append(alpha.detach().clone())

        lbfgs_iter[0] += 1
        if print_training and lbfgs_iter[0] % 10 == 0:
            print(f"L-BFGS Iter {lbfgs_iter[0]} | Total: {total_loss.item():.4e} | Alpha: {alpha.item():.4f}")
        return total_loss

    optimizer_lbfgs.step(closure)

    # Recompute alpha from the final log_alpha now that L-BFGS is done.
    # Without this, alpha would still be whatever it was at the end of the
    # Adam loop above -- the closure's alpha is scoped to the closure only
    # (see comment there) and never updates this name, so every L-BFGS
    # step would otherwise be silently ignored by everything below
    # (the diagnostic, the final print, and the returned "alpha" value).
    alpha = torch.exp(log_alpha)

    # --- DIAGNOSTIC (new): flag suspiciously early L-BFGS termination ---
    # PyTorch's LBFGS can legitimately stop in a handful of calls if
    # tolerance_grad/tolerance_change are satisfied immediately, but a
    # single-digit call count after a long Adam phase is worth a second
    # look rather than silently trusting the result.
    if lbfgs_iter[0] < 10:
        grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        # log_alpha, not alpha: alpha here was just recomputed fresh above
        # and never participated in a backward() call itself, so it has no
        # .grad populated -- log_alpha is the actual leaf Parameter that
        # accumulates gradients.
        if log_alpha.grad is not None:
            grads.append(log_alpha.grad.flatten())
        final_grad_norm = torch.norm(torch.cat(grads)).item() if grads else float('nan')
        print(f"[warning] L-BFGS stopped after only {lbfgs_iter[0]} closure call(s) "
              f"(configured for up to {inverse_config['lbfgs_iters']}). "
              f"Final gradient norm: {final_grad_norm:.4e}. "
              f"This may indicate early convergence (fine) or a degenerate step (worth checking).")

    inverse_train_time = time.time() - inverse_train_start

    # Convert the accumulated per-iteration tensors to plain floats once,
    # in bulk, now that training is done -- a single CPU<->GPU sync
    # instead of one every iteration.
    history_total = torch.stack(history_total).cpu().tolist()
    history_pde = torch.stack(history_pde).cpu().tolist()
    history_bc = torch.stack(history_bc).cpu().tolist()
    history_ic = torch.stack(history_ic).cpu().tolist()
    history_data = torch.stack(history_data).cpu().tolist()
    history_alpha = torch.stack(history_alpha).cpu().tolist()

    print(f"\nTotal training time: {inverse_train_time:.2f}s")
    print(f"Recovered alpha: {alpha.item():.6f} | True alpha: {inverse_config['true_alpha']} | Error: {abs(alpha.item() - inverse_config['true_alpha']):.6f}")

    return {
        "model": model,
        "alpha": alpha.item(),
        "history_total": history_total,
        "history_pde": history_pde,
        "history_bc": history_bc,
        "history_ic": history_ic,
        "history_data": history_data,
        "history_alpha": history_alpha,
        "adam_iters": adam_iters,
        "lbfgs_calls": lbfgs_iter[0],
        "alpha_error": abs(alpha.item() - inverse_config["true_alpha"]),
        "train_time": inverse_train_time,
        "x_obs": x_obs,
        "t_obs": t_obs,
        "u_obs": u_obs,
    }

def fd_solver( N_x, N_t, alpha = 0.4, compute_error = True):
    dx = 1 / (N_x - 1)
    dt = 1 / (N_t - 1)
    
    r = alpha * dt / (dx **2)
    
    x = np.linspace(0, 1, N_x)
    t = np.linspace(0, 1, N_t)
    
    u = np.zeros((N_x, N_t))
    
    u[:, 0] = np.sin(np.pi * x)
    
    #setup tridiagonal matrix (sparse, factored once, reused every timestep)
    main_diag = (1 + r) * np.ones(N_x - 2) # -2 here because endpoints are zeros
    off_diag = (-r/2) * np.ones(N_x - 3)

    A = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csc')
    A_factored = splu(A)

    for  n in range(N_t - 1):
        b = (r/2) * u[:-2, n] + (1-r) * u[1:-1, n] + (r/2) * u[2:, n]
        u_new = A_factored.solve(b)
        u[1:-1, n+1] = u_new
    
    if compute_error:
        X, T = np.meshgrid(x, t, indexing='ij')
        u_exact = np.sin(np.pi * X) * np.exp(-alpha * (np.pi**2) * T)
        error = np.linalg.norm(u - u_exact) / np.linalg.norm(u_exact)
    else:
        error = None

    return x, t, u, error


from scipy.interpolate import RegularGridInterpolator

def cn_nls_baseline(inverse_config, x_obs, t_obs, u_obs):
    
    alpha_candidates = np.linspace(0.01, 1.0, inverse_config.get("n_alpha_candidates", 200))
    
    mse_history = []
    
    cn_nls_start = time.time()  # NEW: added timing (original had none)
    
    for alpha_guess in alpha_candidates:
        x, t, u_guess, _ = fd_solver( N_x = 200, N_t = 200, alpha = alpha_guess, compute_error = False)
        
        interp = RegularGridInterpolator((x, t), u_guess)
        
        obs_points = np.column_stack([
            x_obs.cpu().numpy().flatten(),
            t_obs.cpu().numpy().flatten()
            ])

        u_predicted = interp(obs_points)

        mse = np.mean((u_predicted - u_obs.cpu().numpy().flatten())**2)
        
        mse_history.append(mse)
    
    cn_nls_time = time.time() - cn_nls_start  # NEW
    
    best_mse_index = np.argmin(mse_history)
    
    best_alpha_guess = alpha_candidates[best_mse_index]
    
    print(f"CN-NLS total runtime: {cn_nls_time:.2f}s")  # NEW
    
    return best_alpha_guess, mse_history, alpha_candidates, cn_nls_time


    