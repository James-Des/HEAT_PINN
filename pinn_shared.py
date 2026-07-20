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

fd_solver(), pinn_architecture, sample_points, generate_noisy_data, and
compute_loss_inverse are byte-for-byte what was already in
heat_pinn_tuned.ipynb -- no reformatting, no renamed variables.
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

def train_inverse(inverse_config, print_training=True, trial=None):
    activation  = inverse_config.get("activation",   "tanh")
    lambda_pde  = inverse_config.get("lambda_pde",   1.0)
    lambda_bc   = inverse_config.get("lambda_bc",    1.0)
    lambda_ic   = inverse_config.get("lambda_ic",    1.0)
    lambda_data = inverse_config.get("lambda_data",  1.0)

    model = pinn_architecture(inverse_config["hidden_size"], inverse_config["n_layers"], activation)
    alpha = nn.Parameter(torch.tensor(inverse_config["alpha_init"]))
    optimizer = torch.optim.Adam(list(model.parameters()) + [alpha], lr=inverse_config["adam_lr"])

    x_f, t_f, x_bc, t_bc, x_ic, t_ic = sample_points(inverse_config["N_f"], inverse_config["N_bc"], inverse_config["N_ic"])
    x_obs, t_obs, u_obs = generate_noisy_data(inverse_config)

    history_total = []
    history_pde = []
    history_bc = []
    history_ic = []
    history_data = []
    history_alpha = []

    inverse_train_start = time.time()

    for i in range(1, inverse_config["adam_iters"] + 1):
        optimizer.zero_grad()
        total_loss, pde_loss, bc_loss, ic_loss, data_loss = compute_loss_inverse(
            model, alpha, x_f, t_f, x_bc, t_bc, x_ic, t_ic, x_obs, t_obs, u_obs,
            lambda_pde, lambda_bc, lambda_ic, lambda_data)
        total_loss.backward()
        optimizer.step()

        history_total.append(total_loss.item())
        history_pde.append(pde_loss.item())
        history_bc.append(bc_loss.item())
        history_ic.append(ic_loss.item())
        history_data.append(data_loss.item())
        history_alpha.append(alpha.item())

        if print_training and i % 200 == 0:
            print(f"Iter {i} | Total: {total_loss.item():.4e} | PDE: {pde_loss.item():.4e} | BC: {bc_loss.item():.4e} | IC: {ic_loss.item():.4e} | Alpha: {alpha.item():.4f}")

        if trial is not None and i % 200 == 0:
            import optuna  # local import: only needed on this path, and must be
                            # bound in this module now that train_inverse no longer
                            # lives in the same notebook namespace as `import optuna`
            trial.report(total_loss.item(), i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    adam_iters = len(history_total)

    optimizer_lbfgs = torch.optim.LBFGS(list(model.parameters()) + [alpha], lr=1.0, max_iter=inverse_config["lbfgs_iters"])
    lbfgs_iter = [0]

    def closure():
        optimizer_lbfgs.zero_grad()
        total_loss, pde_loss, bc_loss, ic_loss, data_loss = compute_loss_inverse(
            model, alpha, x_f, t_f, x_bc, t_bc, x_ic, t_ic, x_obs, t_obs, u_obs,
            lambda_pde, lambda_bc, lambda_ic, lambda_data)
        total_loss.backward()

        history_total.append(total_loss.item())
        history_pde.append(pde_loss.item())
        history_bc.append(bc_loss.item())
        history_ic.append(ic_loss.item())
        history_data.append(data_loss.item())
        history_alpha.append(alpha.item())

        lbfgs_iter[0] += 1
        if print_training and lbfgs_iter[0] % 10 == 0:
            print(f"L-BFGS Iter {lbfgs_iter[0]} | Total: {total_loss.item():.4e} | Alpha: {alpha.item():.4f}")
        return total_loss

    optimizer_lbfgs.step(closure)

    # --- DIAGNOSTIC (new): flag suspiciously early L-BFGS termination ---
    # PyTorch's LBFGS can legitimately stop in a handful of calls if
    # tolerance_grad/tolerance_change are satisfied immediately, but a
    # single-digit call count after a long Adam phase is worth a second
    # look rather than silently trusting the result.
    if lbfgs_iter[0] < 10:
        grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        if alpha.grad is not None:
            grads.append(alpha.grad.flatten())
        final_grad_norm = torch.norm(torch.cat(grads)).item() if grads else float('nan')
        print(f"[warning] L-BFGS stopped after only {lbfgs_iter[0]} closure call(s) "
              f"(configured for up to {inverse_config['lbfgs_iters']}). "
              f"Final gradient norm: {final_grad_norm:.4e}. "
              f"This may indicate early convergence (fine) or a degenerate step (worth checking).")

    inverse_train_time = time.time() - inverse_train_start
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

def fd_solver( N_x, N_t, alpha = 0.4, T = 1.0, L = 1, compute_error = True):
    dx = L / (N_x - 1)
    dt = T / (N_t - 1)
    
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
    
    alpha_candidates = np.linspace(0.01, 1.0, 200)
    
    mse_history = []
    
    cn_nls_start = time.time()  # NEW: added timing (original had none)
    
    for alpha_guess in alpha_candidates:
        x, t, u_guess, _ = fd_solver( N_x = 200, N_t = 200, alpha = alpha_guess, compute_error = False)
        
        interp = RegularGridInterpolator((x, t), u_guess)
        
        obs_points = np.column_stack([
            x_obs.numpy().flatten(),
            t_obs.numpy().flatten()
            ])
        
        u_predicted = interp(obs_points)
        
        mse = np.mean((u_predicted - u_obs.numpy().flatten())**2)
        
        mse_history.append(mse)
    
    cn_nls_time = time.time() - cn_nls_start  # NEW
    
    best_mse_index = np.argmin(mse_history)
    
    best_alpha_guess = alpha_candidates[best_mse_index]
    
    print(f"CN-NLS total runtime: {cn_nls_time:.2f}s")  # NEW
    
    return best_alpha_guess, mse_history, alpha_candidates, cn_nls_time


    