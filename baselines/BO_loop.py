from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound, LogExpectedImprovement
from botorch.optim import optimize_acqf
from baselines.GP import GP_Wrapper, GP_MAP_Wrapper, Vanilla_GP_Wrapper
from data import *
import time
from infras.randutils import *
import pickle
from typing import Dict, List
import pandas as pd
from pathlib import Path


def _extract_hyperparams(model) -> Dict[str, float]:
    """Extracts hyperparameters from a trained GP model"""
    params = {}
    
    # Get kernel parameters
    if hasattr(model.gp_model.covar_module, 'base_kernel'):
        kernel = model.gp_model.covar_module.base_kernel
    else:
        kernel = model.gp_model.covar_module
        
    # Lengthscales (ARD or single)
    if hasattr(kernel, 'lengthscale'):
        ls = kernel.lengthscale.detach().cpu().numpy()
        if ls.shape[1] > 1:  # ARD case
            for i in range(ls.shape[1]):
                params[f'lengthscale_dim_{i}'] = float(ls[0, i])
        else:
            params['lengthscale'] = float(ls[0, 0])
    
    # Output scale
    if hasattr(model.gp_model.covar_module, 'outputscale'):
        params['outputscale'] = float(model.gp_model.covar_module.outputscale.detach().cpu().numpy())
    
    # Noise variance
    if hasattr(model.gp_model.likelihood, 'noise'):
        params['noise'] = float(model.gp_model.likelihood.noise.detach().cpu().numpy())
    
    # Kernel-specific parameters
    if hasattr(kernel, 'nu'):  # Matern nu
        params['nu'] = float(kernel.nu)
    elif hasattr(kernel, 'alpha'):  # GeneralCauchy alpha
        params['alpha'] = float(kernel.alpha)
        params['beta'] = float(kernel.beta)
    
    return params


def BO_loop_GP(func_name, dataset, seed, num_step=200, beta=1.5, if_ard=False, if_softplus=True, acqf_type="UCB", set_ls=False,
               kernel_type="mat52",
               device="cpu"):
    #initial storing
    hyperparam_history = []
    best_y = []
    time_list = []
    dim = dataset.func.dims
    bounds = torch.tensor([[0.0] * dim, [1.0] * dim], device=device)

    #create output dir structure
    base_dir = Path(__file__).parent.parent  # Goes up from baselines/ to project root
    output_dir = base_dir / "hyperparams" / func_name / kernel_type
    output_dir.mkdir(parents=True, exist_ok=True)


    model = None  # initialized on i == 1

    for i in range(1, num_step + 1):
        t0 = time.time()

        # full (normalized) dataset every iter
        X, Y = dataset.get_data(normalize=True)
        X, Y = X.to(device), Y.to(device)

        best_y_before = dataset.get_curr_max_unnormed()

        if i == 1:
            model = GP_Wrapper(
                X, Y,
                kernel=kernel_type,
                if_ard=if_ard,
                if_softplus=if_softplus,
                set_ls=set_ls,
                device=device,
            )
            model.init_optimizer(lr=0.1, optim="ADAM")
            model.step(epochs=2000)  # long train once
        else:
            # warm restart: reuse params & optimizer states
            model.update_train_data(X, Y)
            model.step(epochs=50)

        #store hyperparameters after training
        hyperparams = _extract_hyperparams(model)
        hyperparams['iteration'] = i
        hyperparam_history.append(hyperparams)

        # *** Switch to eval() before acqf ***
        model.gp_model.eval()
        model.likelihood.eval()

        if acqf_type == "UCB":
            acqf = UpperConfidenceBound(model=model.gp_model, beta=beta, maximize=True).to(device)
        elif acqf_type == "EI":
            acqf = ExpectedImprovement(model=model.gp_model, best_f=Y.max()).to(device)
        elif acqf_type == "LogEI":
            acqf = LogExpectedImprovement(model=model.gp_model, best_f=Y.max()).to(device)
        else:
            raise NotImplementedError

        try:
            new_x, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=1000,
                options={},
            )
            new_x = new_x.squeeze(0)  # (d,)
        except Exception as e:
            print(f"ERROR during opt acqf ({e}); using random point")
            new_x = torch.rand(dim, device=device)

        dataset.add(new_x.detach().cpu())  # if dataset is on CPU

        time_used = time.time() - t0
        time_list.append(time_used)

        best_y_after = dataset.get_curr_max_unnormed()
        itr = dataset.X.shape[0]
        print(
            f"Seed: {seed} --- Kernel: {kernel_type} --- itr: {itr}: best before={best_y_before}, "
            f"best after={best_y_after}, curr query: {dataset.y[-1]}, "
            f"time={time_used:.3f}s",
            flush=True,
        )
        best_y.append(best_y_before)


        #save hyperparam history
        df_hyperparams = pd.DataFrame(hyperparam_history)
        output_path = output_dir / f"hyperparams_{func_name}_{kernel_type}_seed{seed}.pkl"
        df_hyperparams.to_pickle(output_path)
        print(f"Saved hyperparameters to {output_path}")

    return best_y, time_list



def Vanilla_BO_loop(func_name, dataset, seed, num_step=200):
    """
    Our implementation for Vanilla BO
    """
    best_y = []
    time_list = []
    dim = dataset.func.dims
    for i in range(1, num_step + 1):
        start_time = time.time()
        X, Y = dataset.get_data(normalize=True)
        best_y_before = dataset.get_curr_max_unnormed()
        model = Vanilla_GP_Wrapper(X, Y)
        model.train_model()

        ls = model.gp_model.covar_module.base_kernel.lengthscale
        print(f"ls mean: {ls.mean()}, ls std: {ls.std()}, max: {ls.max()}, min: {ls.min()}")

        acqf = LogExpectedImprovement(model=model.gp_model, best_f=Y.max())
        new_x, _ = optimize_acqf(
            acq_function=acqf,
            bounds=torch.tensor([[0.0] * dim, [1.0] * dim]),
            q=1,
            num_restarts=10,
            raw_samples=1000,
            options={},
        )

        end_time = time.time()
        time_used = end_time - start_time
        time_list.append(time_used)
        dataset.add(new_x)
        best_y_after = dataset.get_curr_max_unnormed()

        print(
            f"Seed: {seed} --- At itr: {i}: best value before={best_y_before}, best value after={best_y_after}, current query: {dataset.y[-1]}",
            flush=True)
        best_y.append(best_y_before)
        
    return best_y, time_list

def BO_loop_GP_MAP(func_name, dataset, seed, num_step=200, beta=1.5, if_ard=True, optim_type="LBFGS", acqf_type="UCB",
                   ls_prior_type="Gamma", set_ls=False, if_matern=False, device="cpu"):
    best_y = []
    time_list = []
    dim = dataset.func.dims
    for i in range(1, num_step + 1):
        start_time = time.time()
        X, Y = dataset.get_data(normalize=True)
        X = X.to(device)
        Y = Y.to(device)
        best_y_before = dataset.get_curr_max_unnormed()
        model = GP_MAP_Wrapper(X, Y, if_ard=if_ard, if_matern=if_matern, optim_type=optim_type,
                               ls_prior_type=ls_prior_type, device=device, set_ls=set_ls)
        model.train_model()

        if acqf_type == "UCB":
            acqf = UpperConfidenceBound(model=model.gp_model, beta=beta, maximize=True).to(device)
        elif acqf_type == "EI":
            acqf = ExpectedImprovement(model=model.gp_model, best_f=Y.max()).to(device)
        else:
            raise NotImplementedError

        new_x, _ = optimize_acqf(
                acq_function=acqf,
                bounds=torch.tensor([[0.0] * dim, [1.0] * dim]),
                q=1,
                num_restarts=10,
                raw_samples=1000,
                options={},
            )

        end_time = time.time()
        time_used = end_time - start_time

        time_list.append(time_used)
        dataset.add(new_x)
        best_y_after = dataset.get_curr_max_unnormed()
        print(
            f"Seed: {seed} --- At itr: {i}: best value before={best_y_before}, best value after={best_y_after}, current query: {dataset.y[-1]}",
            flush=True)
        best_y.append(best_y_before)
    return best_y, time_list