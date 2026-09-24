"""
Core implementation for the heat-equation PINN vs. Crank-Nicolson study.

The problem is the 1D heat equation u_t = alpha * u_xx on the unit domain,
with u(0, t) = u(1, t) = 0 and u(x, 0) = sin(pi * x). It has a known exact
solution, u(x, t) = sin(pi * x) * exp(-alpha * pi^2 * t), which every
accuracy number in the study is measured against.

Two problems are studied:
  Forward: alpha is known, solve for the temperature field u.
  Inverse: alpha is unknown, recover it from sparse, noisy measurements.

Each is solved two ways, a neural network and a classical solver:
  PINN:           train_forward, train_inverse
  Crank-Nicolson: fd_solver, cn_nls_baseline

heat_eqn_pinn.ipynb drives all of it: the Optuna hyperparameter search, the
baseline-vs-tuned PINN comparisons, and the final accuracy and cost tables.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags
from scipy.sparse.linalg import splu


def set_seed(seed: int) -> None:
    """Seed torch and numpy. Call before any run whose numbers get reported."""
    torch.manual_seed(seed)
    np.random.seed(seed)

class pinn_architecture(nn.Module):
    """Fully connected network mapping (x, t) to temperature u."""

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
    """Draw random collocation, boundary, and initial-condition points.

    N_bc is per boundary (x=0 and x=1 each get N_bc), so the total number
    of boundary points returned is 2 * N_bc. N_ic is not doubled.
    """
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
    """Sample the exact solution at random points and add Gaussian noise.

    These are the synthetic measurements the inverse problem recovers
    alpha from.
    """
    x_obs = torch.rand(inverse_config["N_obs"], 1)
    t_obs = torch.rand(inverse_config["N_obs"], 1)
    
    u_exact = torch.sin(torch.pi * x_obs) * torch.exp(-inverse_config["true_alpha"] * torch.pi**2 * t_obs)
    
    noise = torch.randn(inverse_config["N_obs"], 1) * inverse_config["noise_std"]
    
    u_obs = u_exact + noise
    
    return x_obs, t_obs, u_obs

def compute_loss(model, alpha, x_f, t_f, x_bc, t_bc, x_ic, t_ic,
                 lambda_pde=1.0, lambda_bc=1.0, lambda_ic=1.0):
    """Weighted forward loss: PDE residual, boundary, and initial condition.

    Returns (total, pde, bc, ic) so each term can be tracked separately.
    """
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
    """Train a PINN on the forward problem, with alpha known.

    Adam first, then L-BFGS to refine. Returns the trained model, per-term
    loss histories, training time, and relative L2 error on the validation
    grid. Pass a trial to enable Optuna pruning.
    """
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

        # Keep losses as GPU tensors here. Calling .item() every iteration
        # would force a CPU/GPU sync and stall training; they get converted
        # to floats in bulk once training finishes.
        history_total.append(total_loss.detach())
        history_pde.append(pde_loss.detach())
        history_bc.append(bc_loss.detach())
        history_ic.append(ic_loss.detach())

        if print_training and i % 200 == 0:
            print(f"Iter {i} | Total: {total_loss.item():.4e} | PDE: {pde_loss.item():.4e} | BC: {bc_loss.item():.4e} | IC: {ic_loss.item():.4e}")

        if trial is not None and i % 200 == 0:
            import optuna  # only needed on the pruning path
            # Prune on the unweighted PDE/BC/IC residual, not the weighted
            # total loss: lambda_bc and lambda_ic are searched per trial over
            # a wide range, so weighted losses are not comparable between
            # trials. Averaged over the last 200 iterations so a single noisy
            # reading cannot prune a trial that is only momentarily behind.
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

    # Convert the accumulated loss tensors to plain floats in bulk now that
    # training is done, a single CPU/GPU sync instead of one per iteration.
    history_total = torch.stack(history_total).cpu().tolist()
    history_pde = torch.stack(history_pde).cpu().tolist()
    history_bc = torch.stack(history_bc).cpu().tolist()
    history_ic = torch.stack(history_ic).cpu().tolist()

    # Validation grid used for hyperparameter selection, offset by half a
    # cell from the test grid the notebook reports final accuracy on. Without
    # the offset the same points would both pick the winning model and score
    # it, which makes the reported accuracy look better than it is.
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
    """Forward loss plus a data term measuring fit to the noisy observations.

    The data term is what makes alpha identifiable; the physics terms alone
    can be satisfied by the wrong alpha.
    """
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
    """Train a PINN on the inverse problem, recovering alpha from observations.

    Learns the network weights and alpha jointly. Returns the recovered
    alpha, its error against true_alpha, loss and alpha histories, and
    training time. Pass a trial to enable Optuna pruning.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    activation  = inverse_config.get("activation",   "tanh")
    lambda_pde  = inverse_config.get("lambda_pde",   1.0)
    lambda_bc   = inverse_config.get("lambda_bc",    1.0)
    lambda_ic   = inverse_config.get("lambda_ic",    1.0)
    lambda_data = inverse_config.get("lambda_data",  1.0)

    model = pinn_architecture(inverse_config["hidden_size"], inverse_config["n_layers"], activation).to(device)
    # Optimize log_alpha rather than alpha directly, so the recovered
    # diffusivity exp(log_alpha) stays strictly positive no matter what the
    # optimizer does. alpha_init is still the physical guess and must be > 0.
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

        # Same reason as the forward loop: keep these as GPU tensors and
        # convert in bulk later. alpha keeps a .clone() defensively, from
        # when it was a Parameter the optimizer updated in place.
        history_total.append(total_loss.detach())
        history_pde.append(pde_loss.detach())
        history_bc.append(bc_loss.detach())
        history_ic.append(ic_loss.detach())
        history_data.append(data_loss.detach())
        history_alpha.append(alpha.detach().clone())

        if print_training and i % 200 == 0:
            print(f"Iter {i} | Total: {total_loss.item():.4e} | PDE: {pde_loss.item():.4e} | BC: {bc_loss.item():.4e} | IC: {ic_loss.item():.4e} | Alpha: {alpha.item():.4f}")

        if trial is not None and i % 200 == 0:
            import optuna  # only needed on the pruning path
            # Prune on alpha error, the same quantity the final trial
            # selection uses. The residual alone is not enough here: a
            # flexible network can satisfy the PDE for the wrong alpha, and
            # only the data term disambiguates it. Averaged over the last 200
            # iterations because alpha does not converge monotonically.
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
        # Local to this closure. A plain assignment inside a nested function
        # creates its own binding, so this does not update the outer alpha;
        # that gets recomputed after L-BFGS finishes.
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

    # Recompute alpha now that L-BFGS is done. The closure's alpha is local
    # to the closure, so without this every L-BFGS update would be silently
    # dropped from the returned result.
    alpha = torch.exp(log_alpha)

    # Flag suspiciously early L-BFGS termination. PyTorch's LBFGS can
    # legitimately stop in a handful of calls if its tolerances are met
    # immediately, but a single-digit call count after a long Adam phase is
    # worth a second look rather than trusting silently.
    if lbfgs_iter[0] < 10:
        grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        # log_alpha, not alpha: alpha was just recomputed and never went
        # through backward(), so only log_alpha carries a gradient.
        if log_alpha.grad is not None:
            grads.append(log_alpha.grad.flatten())
        final_grad_norm = torch.norm(torch.cat(grads)).item() if grads else float('nan')
        print(f"[warning] L-BFGS stopped after only {lbfgs_iter[0]} closure call(s) "
              f"(configured for up to {inverse_config['lbfgs_iters']}). "
              f"Final gradient norm: {final_grad_norm:.4e}. "
              f"This may indicate early convergence (fine) or a degenerate step (worth checking).")

    inverse_train_time = time.time() - inverse_train_start

    # Convert the accumulated loss tensors to plain floats in bulk now that
    # training is done, a single CPU/GPU sync instead of one per iteration.
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
    """Crank-Nicolson solver for the forward problem.

    Second-order accurate in space and time. The tridiagonal system is
    factored once and reused across all timesteps. Returns (x, t, u, error),
    where error is None if compute_error is False.
    """
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


def cn_nls_baseline(inverse_config, x_obs, t_obs, u_obs):
    """Classical inverse estimator: recover alpha by brute-force search.

    Runs fd_solver for each candidate alpha, interpolates onto the
    observation points, and keeps whichever candidate minimizes mean squared
    error against the observations. Returns the best alpha, the full MSE
    curve, the candidates tried, and total runtime.
    """
    alpha_candidates = np.linspace(0.01, 1.0, inverse_config.get("n_alpha_candidates", 200))
    
    mse_history = []
    
    cn_nls_start = time.time()
    
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
    
    cn_nls_time = time.time() - cn_nls_start
    
    best_mse_index = np.argmin(mse_history)
    
    best_alpha_guess = alpha_candidates[best_mse_index]
    
    print(f"CN-NLS total runtime: {cn_nls_time:.2f}s")
    
    return best_alpha_guess, mse_history, alpha_candidates, cn_nls_time
