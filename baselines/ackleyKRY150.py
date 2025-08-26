# rbrock50.py
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # if using OpenBLAS

import csv
import random
import time
import argparse
from itertools import product
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal

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
FUNC_NAME = "AckleyKRY150"
FUNC = FuncAckleyKRY

# ----- per‐kernel flag & base‐kernel settings -----
def make_kernel_flags():
    kernels    = ["mat12", "mat32", "mat52", "rq", "gcauchy"]
    #kernels    = ["gcauchy"]
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

def already_done(root_dir, func_name, kernel_key, seed):
    subdir = os.path.join(root_dir, func_name, kernel_key)
    fn     = f"{func_name}_{kernel_key}_seed{seed}.csv"
    path   = os.path.join(subdir, fn)
    return os.path.exists(path) and os.path.getsize(path) > 0


def _worker_init(torch_threads=1, interop_threads=1):
    # workers ignore SIGINT; parent will terminate them
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(interop_threads)


def run_one(kernel: str, seed: int, num_iter: int, beta: float,
            root_dir: str, func_name: str = FUNC_NAME,
            skip_existing: bool = False):

    subdir, path = result_path(root_dir, func_name, kernel, seed)
    os.makedirs(subdir, exist_ok=True)
    lock_path = path + ".lock"
    tmp_path  = path + ".part"

    # ---- claim or back off ----
    if skip_existing:
        if os.path.exists(path):
            return 0
        # try to create a lock; if it exists, someone else is doing this job
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            return 0

    try:
        # ---- RNG seeding ----
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # ---- do the work ----
        func    = FuncAckleyKRY(DIM, maximize=True)
        dataset = BayesOptDataset(func, NUM_INIT, 'lhs', seed)

        flags     = KERNEL_FLAGS[kernel]
        base_kern = flags["kernel"]
        set_ls    = 'true' if flags.get("set_ls", False) is True else flags.get("set_ls", False)
        if_ard    = flags.get("if_ard", False)
        noise     = flags.get("noise", None)
        outsc     = flags.get("outscale", None)

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

        # ---- write atomically ----
        with open(tmp_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iteration", "best_obj_val"])
            w.writerows((i, float(v)) for i, v in enumerate(best_vals, 1))
        os.replace(tmp_path, path)  # atomic on POSIX/Windows

        return len(best_vals)

    finally:
        # best-effort cleanup
        try: os.remove(lock_path)
        except FileNotFoundError: pass
        try: os.remove(tmp_path)
        except FileNotFoundError: pass


def result_path(root_dir: str, func_name: str, kernel: str, seed: int):
    subdir = os.path.join(root_dir, func_name, kernel)
    fn = f"{func_name}_{kernel}_seed{seed}.csv"
    return subdir, os.path.join(subdir, fn)

def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("true","t","1","yes","y"):  return True
    if v in ("false","f","0","no","n"):  return False
    raise argparse.ArgumentTypeError("expected true/false")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-workers",   type=int, default=os.cpu_count()-1 or 1)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--num-iter",    type=int, default=NUM_ITER)
    p.add_argument("--func-name",   type=str, default=FUNC_NAME)
    p.add_argument(
        "--skip-existing",
        type=str2bool,
        nargs="?",        # allow no value ⇒ const
        const=True,
        default=True,
        help="skip jobs whose CSV already exists (true/false)",
    )
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    all_tasks = list(product(KERNELS, SEEDS))
    tasks = [(k, s) for (k, s) in all_tasks
             if not already_done(args.results_dir, args.func_name, k, s)]
    if not tasks:
        print("Nothing to do; all outputs present."); return

    max_w = min(args.n_workers, len(tasks))
    print(f"Using {max_w} workers for {len(tasks)} jobs")

    ctx = mp.get_context("spawn")
    start = time.time(); total = 0

    executor = None
    futures = {}
    try:
        executor = ProcessPoolExecutor(
            max_workers=max_w,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(1, 1),
        )
        for (k, s) in tasks:
            fut = executor.submit(
                run_one, k, s, args.num_iter, BETA,
                args.results_dir, args.func_name, args.skip_existing
            )
            futures[fut] = (k, s)

        for fut in as_completed(futures):
            k, s = futures[fut]
            n = fut.result()  # may raise; caught by outer except
            total += n
            print(f"Done {k}, seed={s}: wrote {n} rows")

    except KeyboardInterrupt:
        print("Interrupted: cancelling pending tasks …", flush=True)
        for f in futures: f.cancel()
        if executor is not None:
            # stop queueing, don’t wait; request worker termination
            executor.shutdown(wait=False, cancel_futures=True)
            try:
                for p in getattr(executor, "_processes", {}).values():
                    p.terminate()
            except Exception:
                pass
        raise
    finally:
        if executor is not None:
            # in normal completion this is idempotent; after Ctrl-C, ensures cleanup
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        elapsed = (time.time() - start)/60
        print(f"All finished in {elapsed:.2f} min. Wrote ~{total} rows to {args.results_dir}")

if __name__ == "__main__":
    main()
