# rbrock50.py
import os
import csv
import random
import time
import argparse
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
torch.set_default_dtype(torch.float64)
import gpytorch
import numpy as np

from data import BayesOptDataset, FuncAckleyKRY
from .BO_loop import BO_loop_GP

# ----- globals you likely want to tweak -----
DIM         = 150
NUM_INIT    = 20
NUM_ITER    = 400
BETA        = 1.5
SEEDS       = list(range(0, 10))
DEVICE      = torch.device("cpu")

# ----- per‐kernel flag & base‐kernel settings -----

def make_kernel_flags():
    kernels    = ["mat12", "mat32", "mat52", "rq", "gcauchy"]
    ls_options = [True, "uniform", "lognormal"]
    outscales  = ["hvarfner", "gamma"]
    noise      = "lognormal"

    ls_tag = {True: "ri", "uniform": "unif", "lognormal": "logn"}

    flags = {}
    for k in kernels:
        for ls in ls_options:
            for out in outscales:
                name = f"{k}_{ls_tag[ls]}"
                if out == "gamma":
                    name = f"{name}_gamma"
                flags[name] = {
                    "kernel": k,
                    "if_ard": True,
                    "set_ls": ls,            # True | "uniform" | "lognormal"
                    "noise": noise,          # always "lognormal"
                    "outscale": out,         # "hvarfner" | "gamma"
                    "optim": "LBFGSB",
                }
    return flags

# Use directly:
KERNEL_FLAGS = make_kernel_flags()

KERNELS = list(KERNEL_FLAGS)

def run_one(kernel: str, seed: int, num_iter: int, beta: float,
            root_dir: str, func_name: str = "AckleyKRY150"):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    func    = FuncAckleyKRY(DIM, maximize=True)
    dataset = BayesOptDataset(func, NUM_INIT, 'lhs', seed)

    flags      = KERNEL_FLAGS[kernel]
    base_kern  = flags["kernel"]
    set_ls  = flags.get("set_ls", False)
    if set_ls is True:
        set_ls = 'true'
    if_ard  = flags.get("if_ard", False)
    noise   = flags.get("noise", None)       # str or None
    outsc   = flags.get("outscale", None)    # str or None    

    with gpytorch.settings.cholesky_max_tries(100):
        best_vals, _ = BO_loop_GP(
            func_name=func_name,
            dataset=dataset,
            seed=seed,
            num_step=num_iter,
            beta=beta,
            if_ard=if_ard,
            if_softplus=True,
            acqf_type="UCB",
            set_ls=set_ls,
            kernel_type=base_kern,
            full_kernel_name=kernel,
            device=DEVICE,
            noise_var=noise,
            outputscale=outsc,            
        )

    subdir = os.path.join(root_dir, func_name, kernel)
    os.makedirs(subdir, exist_ok=True)
    fn     = f"{func_name}_{kernel}_seed{seed}.csv"
    path   = os.path.join(subdir, fn)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "best_obj_val"])
        writer.writeheader()
        for i, v in enumerate(best_vals, 1):
            writer.writerow({"iteration": i, "best_obj_val": float(v)})

    return len(best_vals)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-workers",   type=int,
                   default=os.cpu_count()-1 or 1,
                   help="max parallel processes")
    p.add_argument("--results-dir", type=str,
                   default="results",
                   help="root output directory")
    p.add_argument("--num-iter",    type=int,
                   default=NUM_ITER,
                   help="BO iterations per (kernel, seed)")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    tasks = list(product(KERNELS, SEEDS))
    max_w = min(args.n_workers, len(tasks))
    print(f"Using {max_w} workers for {len(tasks)} jobs")

    torch.multiprocessing.set_start_method("spawn", force=True)
    start = time.time()
    total = 0

    with ProcessPoolExecutor(max_workers=max_w) as ex:
        futures = {
            ex.submit(run_one, k, s, args.num_iter, BETA, args.results_dir): (k, s)
            for k, s in tasks
        }
        for fut in as_completed(futures):
            k, s = futures[fut]
            try:
                n = fut.result()
                total += n
                print(f"Done {k}, seed={s}: wrote {n} rows")
            except Exception as e:
                print(f"[ERROR] kernel={k} seed={s}: {e}")

    elapsed = (time.time() - start)/60
    print(f"All finished in {elapsed:.2f} min. Wrote ~{total} rows to {args.results_dir}")

if __name__ == "__main__":
    main()
