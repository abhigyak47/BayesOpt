# mopta8.py
import os
import csv
import random
import time
import argparse
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
torch.set_default_dtype(torch.float64)
import numpy as np

from benchmark.mopta8 import MoptaSoftConstraints
from benchmark.real_dataset import RealDataset
from .BO_loop import BO_loop_GP

# ----- globals you likely want to tweak -----
NUM_INIT = 20
NUM_ITER = 300
BETA = 1.5
SEEDS = list(range(1, 11))
#KERNELS = ["rbf", "mat12", "mat52", "lin*mat52", "gcauchy", "poly2", "poly2*mat52", "mat52+const","rq"]
KERNELS = ["gcauchy","mat12","mat52"]

DEVICE = torch.device("cpu")

def run_one(kernel: str, seed: int, num_iter: int, beta: float, root_dir: str, func_name="mopta8"):
    # keep each worker single-threaded to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    func = MoptaSoftConstraints()
    dataset = RealDataset(func, NUM_INIT, 'lhs', seed)

    best_vals, _ = BO_loop_GP(
        func_name="mopta8",
        dataset=dataset,
        seed=seed,
        num_step=num_iter,
        beta=beta,
        if_ard=True,
        if_softplus=True,
        acqf_type="UCB",
        set_ls=False,
        kernel_type=kernel,
        device=DEVICE
    )

    # File path: results/{function}/{kernel}/{function}_{kernel}_seed{seed}.csv
    subdir = os.path.join(root_dir, func_name, kernel)
    os.makedirs(subdir, exist_ok=True)
    filename = f"{func_name}_{kernel}_seed{seed}.csv"
    filepath = os.path.join(subdir, filename)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "best_obj_val"])
        writer.writeheader()
        for itr, val in enumerate(best_vals, 1):
            writer.writerow({
                "iteration": itr,
                "best_obj_val": float(val),
            })
    
    return len(best_vals)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-workers", type=int, default=os.cpu_count()-1 or 1,
                   help="max parallel processes (default: os.cpu_count()-1)")
    p.add_argument("--results-dir", type=str, default="results",
               help="root output directory (default: results/)")
    p.add_argument("--num-iter", type=int, default=NUM_ITER,
                   help="BO iterations per (kernel, seed)")
    return p.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    tasks = list(product(KERNELS, SEEDS))
    max_workers = min(args.n_workers, len(tasks))
    print(f"Using {max_workers} workers for {len(tasks)} jobs")

    start = time.time()

    # safer with torch
    torch.multiprocessing.set_start_method("spawn", force=True)
    total_rows = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(run_one, k, s, args.num_iter, BETA, args.results_dir): (k, s)
            for k, s in tasks
        }
        for fut in as_completed(futures):
            k, s = futures[fut]
            try:
                rows_written = fut.result()
                total_rows += rows_written
                print(f"Completed kernel={k} seed={s} ({rows_written} rows)")
            except Exception as e:
                print(f"[ERROR] kernel={k} seed={s}: {e}")

    elapsed = time.time() - start
    print(f"Done in {elapsed/60:.2f} min. Wrote {total_rows} rows to {args.results_dir}")

if __name__ == "__main__":
    main()