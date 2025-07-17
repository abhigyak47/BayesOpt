from data import *
import pickle
from infras.randutils import *
#------- COMMENT OUT THE BENCHMARKS -------
# from benchmark.rover_function import Rover
# from benchmark.naslib_benchmark import NasBench201
# from benchmark.svm_benchmark import SVMBenchmark
# from benchmark.mopta8 import MoptaSoftConstraints
# from benchmark.real_dataset import RealDataset
from . import BO_loop
from .BO_loop import BO_loop_GP,  BO_loop_GP_MAP, Vanilla_BO_loop
# from benchmark.DNA import DNA_Lasso
import click
from joblib import Parallel, delayed
#from benchmark.MujocoHumanoid import MujocoHumanoid


class Config:
    def __init__(self, func_name, model_name, seed, beta, if_softplus):
        self.func_name = func_name
        self.model_name = model_name
        self.seed = seed
        self.beta = beta
        self.if_softplus = if_softplus

def all_configs():
    config_list = []
    test_functions = ["Ackley", "Rosenbrock_V1"] # Consider two functions for now
    seeds = [0, 1] # Consider 2 seeds

    # Configuration for RBF Kernel
    config_list.append({
        "func_name": test_functions[0],
        "model_name": "GP_RBF_QuickTest", # Call it whatever
        "seed": seeds[0],
        "beta": 1.5,
        "if_softplus": True,
        "num_step": 5, # Very low training iterations
        "if_ard": True,
        "if_matern": False, # Explicitly set to False for RBF
        "set_ls": False,
        "acqf_type": "UCB",
        "kernel_type": "rbf" # Custom parameter to pass to BO_loop_GP and GP_Wrapper
    })

    # Configuration for Polynomial Kernel (degree 4)
    config_list.append({
        "func_name": test_functions[1],
        "model_name": "GP_Polynomial_QuickTest", # Call it whatever
        "seed": seeds[1],
        "beta": 1.5,
        "if_softplus": True,
        "num_step": 5, # Very low training iterations
        "if_ard": False, # Polynomial kernels often don't use ARD in the same way
        "if_matern": False, # Not Matern
        "set_ls": False,
        "acqf_type": "UCB",
        "kernel_type": "polynomial" # Custom parameter for polynomial kernel
    })
    return config_list

# def all_configs():
#     config_list = []
#     for seed in range(0, 10):
#         for model_name in ["GP_ARD", "GP_ARD_setls", "GP_ARD_RBF", "GP_ARD_RBF_setls"]:
#             for func_name in ['mopta08', 'rover', 'nas201', 'dna', 'SVM', 'Ackley', 'Ackley150', 'StybTang_V1',
#                               'Rosenbrock_V1', 'Rosenbrock100_V1', 'Hartmann6']:
#                 for beta in [1.5]:
#                     for if_softplus in [True]:
#                         config = Config(func_name, model_name, seed, beta, if_softplus)
#                         config_list.append(config)
#     return config_list


def get_config(index):
    config_l = all_configs()
    print(f"{index} out of {len(config_l)}", flush=True)
    return config_l[index]

@click.command()
@click.option("--index", type=int, required=True, help="Which grid index to run.")
def main(index):
    cwd = os.getcwd()
    config = get_config(index)

    # Get the values directly
    model_name = config["model_name"]
    SEED = config["seed"]
    func_name = config["func_name"]
    beta = config["beta"]
    if_softplus = config["if_softplus"]
    num_step = config["num_step"] # Get num_step from config for quick test

    # New parameters for BO_loop_GP:
    if_ard = config["if_ard"]
    if_matern = config["if_matern"]
    set_ls = config["set_ls"]
    acqf_type = config["acqf_type"]
    kernel_type = config["kernel_type"] # Retrieve the new kernel_type

# @click.command()
# @click.option("--index", type=int, required=True, help="Which grid index to run.")
# def main(index):
#     cwd = os.getcwd()
#     config = get_config(index)
#     model_name = config.model_name
#     SEED = config.seed
#     func_name = config.func_name
#     beta = config.beta
#     if_softplus = config.if_softplus

    print(f"Running --- {func_name}, SEED={SEED}, model={model_name}")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_tensor_type(torch.DoubleTensor)

    # if func_name == 'mopta08':
    #     num_step = 300
    #     func = MoptaSoftConstraints()
    #     dst = RealDataset(func, 20, 'lhs', SEED)

    # elif func_name == 'rover':
    #     num_step = 300
    #     func = Rover()
    #     dst = RealDataset(func, 20, 'lhs', SEED)

    # elif func_name == 'nas201':
    #     num_step = 300
    #     func = NasBench201()
    #     dst = RealDataset(func, 20, 'lhs', SEED)

    # elif func_name == 'dna':
    #     num_step = 300
    #     func = DNA_Lasso()
    #     dst = RealDataset(func, 20, 'lhs', SEED)

    # elif func_name == 'SVM':
    #     num_step = 800
    #     func = SVMBenchmark()
    #     dst = RealDataset(func, 20, 'lhs', SEED)

    # elif func_name == 'Ackley':
    #     num_step = 400
    #     D = 150
    #     func = FuncAckley(D, maximize=True)
    #     dst = BayesOptDataset(func, 20, 'lhs', SEED)

    if func_name == 'Ackley':
        num_step = 20
        D = 10
        func = FuncAckley(D, maximize=True)
        dst = BayesOptDataset(func, 20, 'lhs', SEED)

    # elif func_name == 'Ackley150':
    #     num_step = 400
    #     D = 300
    #     func = FuncAckley150(D, maximize=True)
    #     dst = BayesOptDataset(func, 20, 'lhs', SEED)

    # elif func_name == 'StybTang_V1':
    #     num_step = 400
    #     D = 200
    #     func = FuncStybTang_V1(D, maximize=True)
    #     dst = BayesOptDataset(func, 20, 'lhs', SEED)

    # elif func_name == 'Rosenbrock_V1':
    #     num_step = 400
    #     D = 100
    #     func = FuncRosenbrock_V1(D, maximize=True)
    #     dst = BayesOptDataset(func, 20, 'lhs', SEED)

    elif func_name == 'Rosenbrock_V1':
        num_step = 20
        D = 10
        func = FuncRosenbrock_V1(D, maximize=True)
        dst = BayesOptDataset(func, 20, 'lhs', SEED)

    # elif func_name == 'Rosenbrock100_V1':
    #     num_step = 400
    #     D = 300
    #     func = FuncRosenbrock100_V1(D, maximize=True)
    #     dst = BayesOptDataset(func, 20, 'lhs', SEED)

    # elif func_name == 'Hartmann6':
    #     num_step = 400
    #     D = 300
    #     func = FuncHartmann6(D, maximize=True)
    #     dst = BayesOptDataset(func, 20, 'lhs', SEED)

    else:
        raise NotImplementedError


    # try:
    #     if model_name == 'GP_ARD':
    #         best_val, time_list = BO_loop_GP(func_name, dst, SEED, num_step=num_step, beta=beta, if_ard=True,
    #                                          if_softplus=if_softplus, acqf_type="UCB", if_matern=True, set_ls=False)
    #     elif model_name == 'GP_ARD_RBF':
    #         best_val, time_list = BO_loop_GP_MAP(func_name, dst, SEED, num_step=num_step, beta=beta, if_ard=True,
    #                                              optim_type="LBFGS", acqf_type="UCB", ls_prior_type="Uniform",
    #                                              if_matern=False, set_ls=False)
    #     elif model_name == "GP_ARD_setls":
    #         best_val, time_list = BO_loop_GP_MAP(func_name, dst, SEED, num_step=num_step, beta=beta,
    #                                              if_ard=True,
    #                                              optim_type="LBFGS", acqf_type="UCB", ls_prior_type="Uniform",
    #                                              set_ls=True, if_matern=True)
    #     elif model_name == "GP_ARD_RBF_setls":
    #         best_val, time_list = BO_loop_GP_MAP(func_name, dst, SEED, num_step=num_step, beta=beta, if_ard=True,
    #                                              optim_type="LBFGS", acqf_type="UCB", ls_prior_type="Uniform",
    #                                              set_ls=True, if_matern=False)
    #     elif model_name == "Vanilla_BO":
    #         best_val, time_list = Vanilla_BO_loop(func_name, dst, SEED, num_step=num_step)
    #     else:
    #         raise NotImplementedError

    #     BO_result = {
    #         "time": time_list,
    #         "X": dst.X,
    #         "Y": dst.y
    #     }

    #     output_dir = os.path.join(cwd, model_name)
    #     os.makedirs(output_dir, exist_ok=True)
    #     output_file = os.path.join(output_dir, f"{model_name}_{func_name}_{SEED}.pickle")

    #     with open(output_file, 'wb') as handle:
    #         pickle.dump(BO_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # except:
    #     print(f"Error in {index}")



    try:
        best_val, time_list = BO_loop_GP(func_name, dst, SEED, num_step=num_step, beta=beta,
                                        if_ard=if_ard, if_softplus=if_softplus,
                                        acqf_type=acqf_type, if_matern=if_matern, set_ls=set_ls,
                                        kernel_type=kernel_type) # Pass the new kernel_type!

        BO_result = {
            "time": time_list,
            "X": dst.X,
            "Y": dst.y,
            "best_values": best_val # Added best_values to output
        }

        output_dir = os.path.join(cwd, "results", model_name) # Consider adding 'results' subdir
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{model_name}_{func_name}_{SEED}.pickle")

        with open(output_file, 'wb') as handle:
            pickle.dump(BO_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Results saved to: {output_file}", flush=True)

    except Exception as e: 
        print(f"Error in running index {index}: {e}", flush=True)


# if __name__ == "__main__":
#     # This is for HPC run
#     main()

#     # Below is for multiprocess run
#     #start = 0
#     #end = 10
#     #Parallel(n_jobs=(end - start))(delayed(main)(index) for index in range(start, end))

if __name__ == "__main__":
    # --- Quick demo??? ---
    configs = all_configs() # Get configs
    for index in range(len(configs)):
        print(f"\n--- Starting run for config index {index} ---")
        cwd = os.getcwd()
        config = configs[index] # Get the specific config by index

        model_name = config["model_name"]
        SEED = config["seed"]
        func_name = config["func_name"]
        beta = config["beta"]
        if_softplus = config["if_softplus"]
        num_step = config["num_step"]

        if_ard = config["if_ard"]
        if_matern = config["if_matern"]
        set_ls = config["set_ls"]
        acqf_type = config["acqf_type"]
        kernel_type = config["kernel_type"]

        print(f"Running --- {func_name}, SEED={SEED}, model={model_name}, kernel={kernel_type}")
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        torch.set_default_tensor_type(torch.DoubleTensor)

        if func_name == 'Ackley':
            D = 10 
            func = FuncAckley(D, maximize=True)
            dst = BayesOptDataset(func, 20, 'lhs', SEED)
        elif func_name == 'Rosenbrock_V1':
            D = 10 
            func = FuncRosenbrock_V1(D, maximize=True)
            dst = BayesOptDataset(func, 20, 'lhs', SEED)
        else:
            print(f"Skipping unknown function: {func_name}")
            continue 

        # --- BO Loop Execution ---
        try:
            best_val, time_list = BO_loop_GP(func_name, dst, SEED, num_step=num_step, beta=beta,
                                            if_ard=if_ard, if_softplus=if_softplus,
                                            acqf_type=acqf_type, if_matern=if_matern, set_ls=set_ls,
                                            kernel_type=kernel_type)

            BO_result = {
                "time": time_list,
                "X": dst.X,
                "Y": dst.y,
                "best_values": best_val
            }

            output_dir = os.path.join(cwd, "results", model_name)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{model_name}_{func_name}_{SEED}.pickle")

            with open(output_file, 'wb') as handle:
                pickle.dump(BO_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Results saved to: {output_file}", flush=True)

        except Exception as e:
            print(f"Error in running config index {index} ({model_name}, {func_name}, {SEED}): {e}", flush=True)
