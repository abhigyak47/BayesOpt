import numpy as np
import torch
import gpytorch
import math
from data import *
import random
from tqdm import tqdm
#import ssl
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.gpytorch import GPyTorchModel
from torch.quasirandom import SobolEngine
from gpytorch.constraints import Positive, Interval
from gpytorch.priors import HalfCauchyPrior, LogNormalPrior, GammaPrior, UniformPrior, MultivariateNormalPrior, NormalPrior
from gpytorch.functions import MaternCovariance

from gpytorch.kernels.kernel import Kernel
from typing import Optional

from infras.randutils import *

from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from pyro.infer.mcmc import NUTS, MCMC
import pyro
from botorch.models.gp_regression import SingleTaskGP

from functools import partial

#------- ADD POLYNOMIAL KERNEL ---------
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, RQKernel, ConstantKernel, PolynomialKernel
import botorch
from gpytorch.functions import RBFCovariance
from gpytorch.settings import trace_mode
from botorch.optim.stopping import ExpMAStoppingCriterion

#torch.manual_seed(0)
#np.random.seed(0)
#random.seed(0)

#Add Wendland and GC

class WendlandKernel(Kernel):
    r"""
    Compactly-supported Wendland kernel (stationary).

    For distance :math:`r = \|x-x'\|/\ell` and parameters :math:`k \in \{0,1,2\}`, 
    :math:`\mu > 0`:

    .. math::
        k(r) = 
        \begin{cases}
            (1-r)_+^\ell, & k=0 \\
            (1-r)_+^{\ell+1}[1+(\ell+1)r], & k=1 \\
            (1-r)_+^{\ell+2}\left[1+(\ell+2)r+\frac{1}{3}(\ell^2+4\ell+3)r^2\right], & k=2
        \end{cases}

    where :math:`\ell = k + 1 + \mu` and :math:`(1-r)_+ = \max(1-r,0)`.
    """
    has_lengthscale = True
    is_stationary = True

    def __init__(self, mu: float = 1.0, k: int = 0, **kwargs):
        super().__init__(**kwargs)
        if k not in {0, 1, 2}:
            raise ValueError("k must be 0, 1, or 2")
        self.k = k
        
        # Register mu parameter with positive constraint
        self.register_parameter(
            name="raw_mu", 
            parameter=torch.nn.Parameter(torch.tensor(mu))
        )
        self.register_constraint("raw_mu", Positive())

    @property
    def mu(self):
        return self.raw_mu_constraint.transform(self.raw_mu)
    
    @mu.setter
    def mu(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mu)
        self.initialize(raw_mu=self.raw_mu_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        r = self.covar_dist(x1_, x2_, diag=diag, square_dist=False, **params)
        support = (1 - r).clamp(min=0)  # (1-r)_+
        ell = self.k + 1 + self.mu
        
        if self.k == 0:
            return support.pow(ell)
        elif self.k == 1:
            return support.pow(ell+1) * (1 + (ell+1)*r)
        elif self.k == 2:
            return support.pow(ell+2) * (
                1 + (ell+2)*r + (ell**2 + 4*ell + 3)/3 * r**2
            )

class GeneralCauchyKernel(Kernel):
    r"""
    Generalized Cauchy kernel (stationary).

    .. math::
        k(r) = \left(1 + r^\alpha\right)^{-\beta/\alpha}, \quad r = \|x-x'\|/\ell

    with parameters:
    - :math:`\alpha \in (0,2]` (controls shape)
    - :math:`\beta > 0` (controls decay)
    - :math:`\ell` (lengthscale)
    """
    has_lengthscale = True
    is_stationary = True

    def __init__(self, alpha_constraint=None, beta_constraint=None, **kwargs):
        super().__init__(**kwargs)
        
        # Alpha constraint (0 < alpha <= 2)
        if alpha_constraint is None:
            alpha_constraint = Interval(1e-5, 2.0)
        self.register_parameter(
            name="raw_alpha", 
            parameter=torch.nn.Parameter(torch.tensor(1.0))
        )
        self.register_constraint("raw_alpha", alpha_constraint)
        
        # Beta constraint (beta > 0)
        if beta_constraint is None:
            beta_constraint = Positive()
        self.register_parameter(
            name="raw_beta", 
            parameter=torch.nn.Parameter(torch.tensor(1.0))
        )
        self.register_constraint("raw_beta", beta_constraint)

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)
    
    @alpha.setter
    def alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))
    
    @property
    def beta(self):
        return self.raw_beta_constraint.transform(self.raw_beta)
    
    @beta.setter
    def beta(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_beta)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        r = self.covar_dist(x1_, x2_, diag=diag, square_dist=False, **params)
        return (1 + r.pow(self.alpha)).pow(-self.beta/self.alpha)


class ProductKernel(gpytorch.kernels.Kernel):
    def __init__(self, kernel1, kernel2, **kwargs):
        super().__init__(**kwargs)
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def forward(self, x1, x2, **params):
        return self.kernel1(x1, x2, **params) * self.kernel2(x1, x2, **params)


KERNEL_DEFAULTS = {
    # Format: KernelType: (constructor, default_nu, supports_ard)
    "mat12": (MaternKernel, 0.5, True),
    "mat32": (MaternKernel, 1.5, True),
    "mat52": (MaternKernel, 2.5, True),
    "rbf": (RBFKernel, None, True),
    "rq": (RQKernel, None, True),

    "lin*mat52": (
        lambda ard_num_dims, **kwargs: ProductKernel(
            gpytorch.kernels.LinearKernel(ard_num_dims=ard_num_dims),
            gpytorch.kernels.MaternKernel(nu=5.5, ard_num_dims=ard_num_dims)
        ),
        None,  # No nu parameter for product kernels
        True   # 
    ),

    "wendland": (WendlandKernel, None, False),
    "gcauchy": (GeneralCauchyKernel, None, False),
    "poly2": (
        lambda ard_num_dims=None, **kwargs: PolynomialKernel(power=2),
        None,
        False
    ),
    "poly2*mat52": (
        lambda ard_num_dims, **kwargs: ProductKernel(
            PolynomialKernel(power=2),
            MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        ),
        None,
        True
    ),
    "mat52+const": (
        lambda ard_num_dims, **kwargs: gpytorch.kernels.AdditiveKernel(
            MaternKernel(nu=2.5, ard_num_dims=ard_num_dims),
            ConstantKernel()
        ),
        None,
        True
    )
}

def inv_sigmoid(x):
    return torch.log(x) - torch.log(1 - x)


class Vanilla_GP_Wrapper:
    def __init__(self, train_x, train_y):
        self.X = train_x
        self.dim = self.X.shape[1]
        self.y = train_y.reshape(-1, 1)

        ls_prior = LogNormalPrior(math.sqrt(2)+(math.log(self.dim)/2.0), math.sqrt(3))
        covar_module = ScaleKernel(
            base_kernel=RBFKernel(
                ard_num_dims=train_x.shape[1],
                nu=2.5,
                lengthscale_prior=ls_prior,
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        self.gp_model = SingleTaskGP(self.X, self.y, covar_module=covar_module)

    def train_model(self):
        self.gp_model.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
        optimizer = torch.optim.RMSprop(mll.parameters(), lr=0.1)
        botorch.fit.fit_gpytorch_mll_torch(mll, optimizer=optimizer)

    def pred(self, test_x, num_samples=8):
        self.gp_model.eval()
        f_pred = self.gp_model(test_x)
        means = f_pred.mean
        vars = f_pred.variance
        dist = torch.distributions.MultivariateNormal(
            means.squeeze(),
            torch.diag(vars.squeeze())
        )
        samples = dist.sample((num_samples,)).permute(1, 0)
        return samples, means, vars


class ExactGPModel(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1
    def __init__(self, train_x, train_y, likelihood, if_ard=True, if_softplus=True, set_ls=False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.ls_constraint = None
        self.D = train_x.shape[1]

        if if_ard:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[1], lengthscale_constraint=self.ls_constraint),
            )
            if set_ls:
                ls = torch.ones_like(self.covar_module.base_kernel.lengthscale) * math.sqrt(self.D)
                self.covar_module.base_kernel._set_lengthscale(ls)
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(lengthscale_constraint=self.ls_constraint))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#------- DEFINE A ClASS FOR THE POLYNOMIAL KERNEL ------
class ExactGPModelPolynomial(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1
    def __init__(self, train_x, train_y, likelihood, ard_num_dims=None, power=4, set_ls=False): # Added power
        super(ExactGPModelPolynomial, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.ls_constraint = None
        self.D = train_x.shape[1]

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PolynomialKernel(power=power), # Use PolynomialKernel with specified power
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGPModelRBF(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1
    def __init__(self, train_x, train_y, likelihood, if_ard=True, if_softplus=True, set_ls=False):
        super(ExactGPModelRBF, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.ls_constraint = None
        self.D = train_x.shape[1]
        if not if_softplus:
            self.ls_constraint = Positive(transform=torch.exp, inv_transform=torch.log)
        if if_ard:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1], lengthscale_constraint=None),
            )
            if set_ls:
                ls = torch.ones_like(self.covar_module.base_kernel.lengthscale) * math.sqrt(self.D)
                self.covar_module.base_kernel._set_lengthscale(ls)
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_constraint=self.ls_constraint))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GP_Wrapper:
    def __init__(self, train_x, train_y, kernel="mat52", if_ard=False, 
                 if_softplus=True, set_ls=False, device="cpu", **kernel_args):
        self.device = device
        self.X = train_x.to(device)
        self.y = train_y.squeeze().to(device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-16, 1e-2, initial_value = 1e-7)).to(device)

        # Handle kernel construction
        if isinstance(kernel, str):
            try:
                kernel_class, default_nu, supports_ard = KERNEL_DEFAULTS[kernel]
                base_kernel = kernel_class(
                    nu=default_nu if kernel_class == MaternKernel else None,
                    ard_num_dims=train_x.shape[1] if (if_ard and supports_ard) else None,
                    **kernel_args
                )
            except KeyError:
                raise ValueError(f"Unknown kernel '{kernel}'. Valid: {list(KERNEL_DEFAULTS.keys())}")
        else:
            base_kernel = kernel  # Direct kernel injection

        # Handle constraints/priors
        if not if_softplus:
            base_kernel.lengthscale_constraint = Positive(transform=torch.exp, inv_transform=torch.log)
        
        if set_ls:
            base_kernel.lengthscale = torch.ones_like(base_kernel.lengthscale) * math.sqrt(train_x.shape[1])

        # Finalize
        self.gp_model = ExactGPModel(
            self.X, self.y, self.likelihood,
            if_ard=if_ard,
            if_softplus=if_softplus,
            set_ls=set_ls
        ).to(device)
        self.gp_model.covar_module = ScaleKernel(base_kernel).to(device)

        # warm-restart state
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model).to(device)
        self.optimizer = None  # created on first call to init_optimizer

# class GP_Wrapper:
#     def __init__(self, train_x, train_y, if_ard=False, if_softplus=True, if_matern=True, set_ls=False, device="cpu"):
#         self.device = device
#         self.X = train_x
#         self.y = train_y.squeeze()

#         self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
#         if if_matern:
#             self.gp_model = ExactGPModel(self.X, self.y, self.likelihood, if_ard, if_softplus, set_ls=set_ls).to(self.device)
#         else:
#             self.gp_model = ExactGPModelRBF(self.X, self.y, self.likelihood, if_ard, if_softplus, set_ls=set_ls).to(self.device)

    def train_model(self, epochs=500, lr=0.1, optim="ADAM"):
        self.gp_model.train()
        self.likelihood.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model).to(self.device)
        if optim == "ADAM":
            optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=lr)
        elif optim == "RMSPROP":
            optimizer = torch.optim.RMSprop(self.gp_model.parameters(), lr=lr)
        elif optim == "botorch":
            stop_c = ExpMAStoppingCriterion(
                maxiter=10000,
                minimize=True,
                n_window=10,
                eta=1.0,
                rel_tol=1e-6,
            )

            optimizer = torch.optim.RMSprop(self.gp_model.parameters(), lr=0.05)
            botorch.fit.fit_gpytorch_mll_torch(mll, optimizer=optimizer, step_limit=1500, stopping_criterion=stop_c)
            return
        else:
            raise NotImplementedError

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.gp_model(self.X)
            loss = -mll(output, self.y)
            loss.backward()
            optimizer.step()

    
    # ---- NEW ----
    def init_optimizer(self, lr=0.1, optim="ADAM"):
        if optim == "ADAM":
            self.optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=lr)
        elif optim == "RMSPROP":
            self.optimizer = torch.optim.RMSprop(self.gp_model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Only ADAM/RMSPROP supported here, got {optim}.")
        return self.optimizer

    # ---- NEW ----
    @torch.enable_grad()
    def step(self, epochs: int):
        self.gp_model.train()
        self.likelihood.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            output = self.gp_model(self.X)
            loss = -self.mll(output, self.y)
            loss.backward()
            self.optimizer.step()

    # ---- NEW ----
    def update_train_data(self, train_x, train_y):
        self.X = train_x.to(self.device)
        self.y = train_y.squeeze().to(self.device)
        # crucial bit: do NOT recreate the model, just point it to the new data
        self.gp_model.set_train_data(inputs=self.X, targets=self.y, strict=False)


    def pred(self, test_x, num_samples=8):
        self.gp_model.eval()
        f_pred = self.gp_model(test_x)
        means = f_pred.mean
        vars = f_pred.variance
        dist = torch.distributions.MultivariateNormal(
            means.squeeze(),
            torch.diag(vars.squeeze())
        )
        samples = dist.sample((num_samples,)).permute(1, 0)
        return samples, means, vars


class GP_MAP_Wrapper:
    def __init__(self, train_x, train_y, if_ard=True, ls_prior_type="Uniform", optim_type="LBFGS", set_ls=False,
                 if_matern=True, device="cpu"):
        self.device = device
        self.X = train_x
        self.D = train_x.shape[1]
        self.y = train_y.reshape(-1, 1)
        self.optim_type = optim_type
        if ls_prior_type == "Gamma":
            ls_prior = GammaPrior(3.0, 6.0)
            ls_constraint = None
        elif ls_prior_type == "Uniform":
            if self.D >= 100:
                ls_prior = UniformPrior(1e-10, 30)
                ls_constraint = Interval(lower_bound=1e-10, upper_bound=30)
            else:
                ls_prior = UniformPrior(1e-10, 10.0)
                ls_constraint = Interval(lower_bound=1e-10, upper_bound=10.0)
        else:
            raise NotImplementedError

        if if_ard and if_matern:
            covar_module = ScaleKernel(
                base_kernel=MaternKernel(ard_num_dims=train_x.shape[1], lengthscale_prior=ls_prior,
                                                    lengthscale_constraint=ls_constraint),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            if set_ls:
                ls = torch.ones_like(covar_module.base_kernel.lengthscale) * math.sqrt(self.D)
                covar_module.base_kernel._set_lengthscale(ls)
            else:
                ls = torch.ones_like(covar_module.base_kernel.lengthscale) * 0.6931
                covar_module.base_kernel._set_lengthscale(ls)

        elif if_ard and not if_matern:
            covar_module = ScaleKernel(
                base_kernel=RBFKernel(ard_num_dims=train_x.shape[1], lengthscale_prior=ls_prior,
                                         lengthscale_constraint=ls_constraint),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            if set_ls:
                ls = torch.ones_like(covar_module.base_kernel.lengthscale) * math.sqrt(self.D)
                covar_module.base_kernel._set_lengthscale(ls)
            else:
                ls = torch.ones_like(covar_module.base_kernel.lengthscale) * 0.6931
                covar_module.base_kernel._set_lengthscale(ls)

        else:
            raise NotImplementedError

        self.gp_model = SingleTaskGP(self.X, self.y, covar_module=covar_module).to(self.device)

    def train_model(self):
        self.gp_model.train()
        if self.optim_type == "LBFGS":
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model).to(self.device)
            botorch.fit.fit_gpytorch_mll(mll)
        elif self.optim_type == "ADAM":
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model).to(self.device)
            stop_c = ExpMAStoppingCriterion(
                maxiter = 10000,
                minimize = True,
                n_window = 10,
                eta = 1.0,
                rel_tol = 1e-6,
                )

            optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=0.05)
            botorch.fit.fit_gpytorch_mll_torch(mll, optimizer=optimizer, step_limit=1000, stopping_criterion=stop_c)

    def pred(self, test_x, num_samples=8):
        self.gp_model.eval()
        f_pred = self.gp_model(test_x)
        means = f_pred.mean
        vars = f_pred.variance
        dist = torch.distributions.MultivariateNormal(
            means.squeeze(),
            torch.diag(vars.squeeze())
        )
        samples = dist.sample((num_samples,)).permute(1, 0)
        return samples, means, vars
