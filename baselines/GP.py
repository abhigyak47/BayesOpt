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
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, RQKernel, PolynomialKernel
import botorch
from gpytorch.functions import RBFCovariance
from gpytorch.settings import trace_mode
from botorch.optim.stopping import ExpMAStoppingCriterion

#torch.manual_seed(0)
#np.random.seed(0)
#random.seed(0)

#Add Wendland and GC

class ModifiedGeneralCauchyKernel(Kernel):
    r"""
    Modified Generalized Cauchy kernel (stationary):

    .. math::
        c(h) = (1 + |h|^\alpha)^{-\frac\beta\alpha - 1}
               \,\Bigl[\,1 + (1 - \beta)\,|h|^\alpha\Bigr]

    with trainable parameters
    - :math:`\alpha\in(0,2]`
    - :math:`\beta>0`
    - lengthscale :math:`\ell`
    """
    has_lengthscale = True
    is_stationary  = True

    def __init__(
        self,
        alpha_constraint=None,
        beta_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        # α constraint: (0,2]
        if alpha_constraint is None:
            alpha_constraint = Interval(1e-5, 2.0)
        self.register_parameter(
            name="raw_alpha",
            parameter=torch.nn.Parameter(torch.tensor(1.0))
        )
        self.register_constraint("raw_alpha", alpha_constraint)

        # β constraint: > 0
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
        self.initialize(
            raw_alpha=self.raw_alpha_constraint.inverse_transform(torch.as_tensor(value))
        )

    @property
    def beta(self):
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value):
        self.initialize(
            raw_beta=self.raw_beta_constraint.inverse_transform(torch.as_tensor(value))
        )

    def forward(self, x1, x2, diag=False, **params):
        # rescale by lengthscale
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)

        # pairwise distance r = ||x1 - x2|| / ℓ
        r = self.covar_dist(x1_, x2_, diag=diag, square_dist=False, **params)

        # base power: (1 + r^α)^(-β/α - 1)
        base = (1 + r.pow(self.alpha)).pow(-self.beta/self.alpha - 1)

        # modification factor: 1 + (1 - β) r^α
        mod = 1 + (1 - self.beta) * r.pow(self.alpha)

        return base * mod

class WendlandKernel(Kernel):
    r"""
    Compactly-supported Wendland kernel (stationary).

    For distance :math:`r = \|x-x'\|/lengthscale` and parameters :math:`k \in \{0,1,2\}`

    .. math::
        k(r) = 
        \begin{cases}
            (1-r)_+^\ell, & k=0 \\
            (1-r)_+^{\ell+1}[1+(\ell+1)r], & k=1 \\
            (1-r)_+^{\ell+2}\left[1+(\ell+2)r+\frac{1}{3}(\ell^2+4\ell+3)r^2\right], & k=2
        \end{cases}

    where :math:`\ell = k + 1` and :math:`(1-r)_+ = \max(1-r,0)`.
    """
    has_lengthscale = True
    is_stationary = True

    def __init__(self, k: int = 0, **kwargs):
        super().__init__(**kwargs)
        if k not in {0, 1, 2}:
            raise ValueError("k must be 0, 1, or 2")
        self.k = k
        
    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        r = self.covar_dist(x1_, x2_, diag=diag, square_dist=False, **params)
        support = (1 - r).clamp(min=0)  # (1-r)_+
        ell = self.k + 1

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

    Parameters:
    - alpha ∈ (0, 2] (controls shape)
    - beta > 0 (controls decay)
    - lengthscale ℓ can be scalar or vector (via `ard_num_dims`)
    """
    has_lengthscale = True
    is_stationary = True

    def __init__(
        self,
        alpha_constraint=None,
        beta_constraint=None,
        ard_num_dims: Optional[int] = None,
        **kwargs
    ):
        super().__init__(ard_num_dims=ard_num_dims, **kwargs)

        if alpha_constraint is None:
            alpha_constraint = Interval(1e-5, 2.0)
        self.register_parameter(
            name="raw_alpha",
            parameter=torch.nn.Parameter(torch.tensor(1.0))
        )
        self.register_constraint("raw_alpha", alpha_constraint)

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
        value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    @property
    def beta(self):
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value):
        value = torch.as_tensor(value).to(self.raw_beta)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1 / self.lengthscale
        x2_ = x2 / self.lengthscale
        r = self.covar_dist(x1_, x2_, diag=diag, square_dist=False, **params)
        return (1 + r.pow(self.alpha)).pow(-self.beta / self.alpha)



import torch
from typing import Optional
from gpytorch.kernels.kernel import Kernel
from gpytorch.constraints import Interval, Positive

class GeneralCauchyKernelAltParameterization(Kernel):
    r"""
    Alternate parameterization of the Generalized Cauchy kernel (stationary):

    .. math::
        k(r) = \bigl(1 + \tfrac{r^\gamma}{2\,\alpha'}\bigr)^{-\alpha'},
        \quad r = \|x - x'\| / \rho

    Parameters:
    - gamma ∈ (0, 2] (shape parameter)
    - alpha_prime > 0 (decay parameter)
    - lengthscale ρ can be scalar or vector (via `ard_num_dims`)

    H = 1-gamma * alpha'/2 \in (1/2, 1) is Hurst parameter when $\alpha' \in (0, 1/\gamma)$.
    """
    has_lengthscale = True
    is_stationary = True

    def __init__(
        self,
        gamma_constraint=None,
        alpha_prime_constraint=None,
        ard_num_dims: Optional[int] = None,
        **kwargs
    ):
        super().__init__(ard_num_dims=ard_num_dims, **kwargs)

        # γ ∈ (0,2]
        if gamma_constraint is None:
            gamma_constraint = Interval(1e-5, 2.0)
        self.register_parameter(
            name="raw_gamma",
            parameter=torch.nn.Parameter(torch.tensor(1.0))
        )
        self.register_constraint("raw_gamma", gamma_constraint)

        # α' > 0
        if alpha_prime_constraint is None:
            alpha_prime_constraint = Positive()
        self.register_parameter(
            name="raw_alpha_prime",
            parameter=torch.nn.Parameter(torch.tensor(0.5))
        )
        self.register_constraint("raw_alpha_prime", alpha_prime_constraint)  

    @property
    def gamma(self):
        return self.raw_gamma_constraint.transform(self.raw_gamma)

    @gamma.setter
    def gamma(self, value):
        value = torch.as_tensor(value).to(self.raw_gamma)
        self.initialize(raw_gamma=self.raw_gamma_constraint.inverse_transform(value))

    @property
    def alpha_prime(self):
        return self.raw_alpha_prime_constraint.transform(self.raw_alpha_prime)

    @alpha_prime.setter
    def alpha_prime(self, value):
        value = torch.as_tensor(value).to(self.raw_alpha_prime)
        self.initialize(raw_alpha_prime=self.raw_alpha_prime_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        # r = ||x - x'|| / ρ
        x1_ = x1 / self.lengthscale
        x2_ = x2 / self.lengthscale
        r = self.covar_dist(x1_, x2_, diag=diag, square_dist=False, **params)
        # k(r) = (1 + r^γ / (2 α'))^{-α'}
        return (1 + r.pow(self.gamma).div(2 * self.alpha_prime)).pow(-self.alpha_prime)



class GeneralCauchyKernelFractal(Kernel):
    r"""
    Generalized Cauchy kernel (stationary).

    .. math::
        k(r) = \left(1 + r^\alpha\right)^{-\beta/\alpha}, \quad r = \|x-x'\|/\ell

    Parameters:
    - alpha ∈ (0, 2] (controls shape)
    - beta > 0 (controls decay)
    - lengthscale ℓ can be scalar or vector (via `ard_num_dims`)
    """
    has_lengthscale = True
    is_stationary = True

    def __init__(
        self,
        alpha_constraint=None,
        beta_constraint=None,
        ard_num_dims: Optional[int] = None,
        **kwargs
    ):
        super().__init__(ard_num_dims=ard_num_dims, **kwargs)

        if alpha_constraint is None:
            alpha_constraint = Interval(1e-5, 2.0)
        self.register_parameter(
            name="raw_alpha",
            parameter=torch.nn.Parameter(torch.tensor(1.0))
        )
        self.register_constraint("raw_alpha", alpha_constraint)

        if beta_constraint is None:
            beta_constraint = Interval(1e-5, 1.0)
        self.register_parameter(
            name="raw_beta",
            parameter=torch.nn.Parameter(torch.tensor(0.5))
        )
        self.register_constraint("raw_beta", beta_constraint)

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    @property
    def beta(self):
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value):
        value = torch.as_tensor(value).to(self.raw_beta)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1 / self.lengthscale
        x2_ = x2 / self.lengthscale
        r = self.covar_dist(x1_, x2_, diag=diag, square_dist=False, **params)
        return (1 + r.pow(self.alpha)).pow(-self.beta / self.alpha)



class FullyARDGeneralCauchyKernel(Kernel):
    has_lengthscale, is_stationary = True, True
    def __init__(self, ard_num_dims: int, alpha_constraint=None, beta_constraint=None, **kwargs):
        kwargs.pop("nu", None); kwargs.pop("k", None)
        ard_num_dims = int(ard_num_dims)
        super().__init__(ard_num_dims=ard_num_dims, **kwargs)

        D = int(self.lengthscale.size(-1))
        if alpha_constraint is None: alpha_constraint = Interval(1e-3, 2.0)
        if beta_constraint  is None: beta_constraint  = Positive()

        # match lengthscale shape: (1, D)
        self.register_parameter("raw_alpha", torch.nn.Parameter(torch.ones(1, D)))
        self.register_constraint("raw_alpha", alpha_constraint)
        self.register_parameter("raw_beta",  torch.nn.Parameter(torch.ones(1, D)))
        self.register_constraint("raw_beta",  beta_constraint)

    @property
    def alpha(self): return self.raw_alpha_constraint.transform(self.raw_alpha)
    @alpha.setter
    def alpha(self, v):
        D = self.raw_alpha.shape[-1]
        v = torch.as_tensor(v, dtype=self.raw_alpha.dtype, device=self.raw_alpha.device)
        if v.numel() == 1:
            v = v.expand_as(self.raw_alpha)
        elif v.shape[-1] == D:
            v = v.view(1, D)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(v))

    @property
    def beta(self): return self.raw_beta_constraint.transform(self.raw_beta)
    @beta.setter
    def beta(self, v):
        D = self.raw_beta.shape[-1]
        v = torch.as_tensor(v, dtype=self.raw_beta.dtype, device=self.raw_beta.device)
        if v.numel() == 1:
            v = v.expand_as(self.raw_beta)
        elif v.shape[-1] == D:
            v = v.view(1, D)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(v))

    def forward(self, x1, x2, diag: bool = False, **params):
        # active dims + ARD scaling
        x1 = x1[..., self.active_dims] if getattr(self, "active_dims", None) is not None else x1
        x2 = x2[..., self.active_dims] if getattr(self, "active_dims", None) is not None else x2
        x1s = x1 / self.lengthscale
        x2s = x2 / self.lengthscale

        if diag:
            r = (x1s - x2s).abs()                   # (N, D)
        else:
            r = (x1s.unsqueeze(1) - x2s.unsqueeze(0)).abs()  # (N, M, D)

        # --- stable per-dim term in log-space ---
        # Avoid log(0): compute r^α as exp(α * log r) only where r>0.
        eps = 1e-12
        r_pos = r > 0
        log_r = torch.zeros_like(r)
        log_r[r_pos] = torch.log(r[r_pos].clamp_min(eps))

        r_alpha = torch.zeros_like(r)
        r_alpha[r_pos] = torch.exp(self.alpha * log_r)[r_pos]          # 0 when r==0 (no grad through log 0)

        log1p_ralpha = torch.log1p(r_alpha)                             # stable near 0
        exponent = -self.beta / self.alpha                              # (1,1,D)
        log_k_per_dim = exponent * log1p_ralpha                         # (..., D)

        # At r==0, contribution is exactly 0; we already have r_alpha=0 ⇒ log1p(0)=0 ⇒ fine.
        # Sum over dimensions (product in log-space), then exp back
        log_k = log_k_per_dim.sum(dim=-1)                               # (...,)
        return torch.exp(log_k)


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
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        ),
        None,  # No nu parameter for product kernels
        True   # 
    ),

    "wendland": (WendlandKernel, 0, True),     # left for legacy purposes
    "wendland0": (WendlandKernel, 0, True),
    "wendland1": (WendlandKernel, 1, True),
    "wendland2": (WendlandKernel, 2, True),
    "gcauchy": (GeneralCauchyKernelAltParameterization, None, True),
    "gcauchyfractal": (GeneralCauchyKernelFractal, None, True),
    #"gcauchyaltparam": (GeneralCauchyKernelAltParameterization, None, True),
    "fullyardgcauchy": (FullyARDGeneralCauchyKernel, None, True),
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
    )
    # "mat52+const": (
    #     lambda ard_num_dims, **kwargs: gpytorch.kernels.AdditiveKernel(
    #         MaternKernel(nu=2.5, ard_num_dims=ard_num_dims),
    #         ConstantKernel()
    #     ),
    #     None,
    #     True
    # )
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
    def __init__(self, train_x, train_y, likelihood, if_ard=True):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        self.D = train_x.shape[1]

        if if_ard:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[1]),
            )
            # if set_ls:
            #     ls = torch.ones_like(self.covar_module.base_kernel.lengthscale) * math.sqrt(self.D)
            #     self.covar_module.base_kernel._set_lengthscale(ls)
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

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

def _lower_or_none(x):
    return x.lower() if isinstance(x, str) else None

class GP_Wrapper:
    def __init__(self, train_x, train_y, kernel="mat52", if_ard=False, 
                 if_softplus=True, set_ls=False, device="cpu", noise_var = None, outputscale = None, **kernel_args):
        self.device = device
        self.X = train_x.to(device)
        self.y = train_y.squeeze().to(device)

        ls_opt = _lower_or_none(set_ls)          # string mode: 'lognormal'/'uniform'/'true'
        init_ls = bool(set_ls is True)           # boolean mode from your KERNEL_FLAGS

        if noise_var is None:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-16, 1e-2, initial_value = 1e-7)).to(device)
        elif noise_var.lower() == "lognormal":
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=LogNormalPrior(-4.0, 1.0)).to(device)
        elif noise_var.lower() == "gamma":
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=GammaPrior(1.1, 0.05)).to(device)

        dim = train_x.shape[1]

        if ls_opt == "lognormal":
            ls_prior = LogNormalPrior(math.sqrt(2) + math.log(dim)/2, math.sqrt(3))
            ls_constraint = None
        elif ls_opt == "uniform":
            ls_prior = UniformPrior(0.0, 30.0)
            ls_constraint = gpytorch.constraints.Interval(1e-10, 30.0)
        else:
            ls_prior, ls_constraint = None, None     

        # Handle kernel construction
        if isinstance(kernel, str):
            try:
                kernel_class, default_nu, supports_ard = KERNEL_DEFAULTS[kernel]
                base_kernel = kernel_class(
                    nu=default_nu if kernel_class == MaternKernel else None,
                    k=default_nu if kernel_class == WendlandKernel else None,
                    ard_num_dims=train_x.shape[1] if (if_ard and supports_ard) else None,
                    lengthscale_prior=ls_prior if ls_opt in ["lognormal", "uniform"] else None,
                    lengthscale_constraint=ls_constraint,
                    **kernel_args
                )
            except KeyError:
                raise ValueError(f"Unknown kernel '{kernel}'. Valid: {list(KERNEL_DEFAULTS.keys())}")
        else:
            base_kernel = kernel  # Direct kernel injection

        if ls_opt == "true" or ls_opt == "uniform":
            base_kernel.lengthscale = torch.ones_like(base_kernel.lengthscale) * math.sqrt(train_x.shape[1])



        # Finalize
        self.gp_model = ExactGPModel(
            self.X, self.y, self.likelihood,
            if_ard=if_ard
        ).to(device)

        if outputscale is None:
            self.gp_model.covar_module = ScaleKernel(base_kernel).to(device)
        elif outputscale.lower() == "hvarfner":
            full_kernel = ScaleKernel(base_kernel).to(device)
            full_kernel.outputscale = 1
            full_kernel.raw_outputscale.requires_grad = False
            self.gp_model.covar_module = full_kernel
        elif outputscale.lower() == "gamma":
            full_kernel = ScaleKernel(
                base_kernel,
                outputscale_prior=GammaPrior(2.0, 2.0),
            ).to(device)
        else:
            # raise error
            raise ValueError(f"Unknown outputscale '{outputscale}'. Valid: [None, 'hvarfner', 'gamma']")

        # warm-restart state
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model).to(device)
        self.optimizer = None  # created on first call to init_optimizer
            
    def train_model(self, epochs=500, lr=0.1, optim="ADAM"):
        self.gp_model.train()
        self.likelihood.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model).to(self.device)

        optu = optim.upper()
        if optu == "ADAM":
            optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=lr)
        elif optu == "RMSPROP":
            optimizer = torch.optim.RMSprop(self.gp_model.parameters(), lr=lr)
        elif optu in {"LBFGSB", "LBFGS-B", "SCIPY"}:
            # True L-BFGS-B via SciPy; honors gpytorch constraints as bounds
            try:
                from botorch.optim.fit import fit_gpytorch_mll_scipy
            except ImportError:
                from botorch.fit import fit_gpytorch_mll_scipy  # older BoTorch
            fit_gpytorch_mll_scipy(mll, options={"maxiter": int(epochs), "disp": False})
            return
        elif optu == "BOTORCH":
            stop_c = ExpMAStoppingCriterion(maxiter=10000, minimize=True, n_window=10, eta=1.0, rel_tol=1e-6)
            optimizer = torch.optim.RMSprop(self.gp_model.parameters(), lr=0.05)
            botorch.fit.fit_gpytorch_mll_torch(mll, optimizer=optimizer, step_limit=1500, stopping_criterion=stop_c)
            return
        else:
            raise NotImplementedError(f"Unknown optim: {optim}")

        # standard PyTorch loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.gp_model(self.X)
            loss = -mll(output, self.y)
            loss.backward()
            optimizer.step()

    
    def init_optimizer(self, lr=0.1, optim="ADAM"):
        optu = optim.upper()
        self._optim_name = optu
        if optu == "ADAM":
            self.optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=lr)
        elif optu == "RMSPROP":
            self.optimizer = torch.optim.RMSprop(self.gp_model.parameters(), lr=lr)
        elif optu in {"LBFGSB", "LBFGS-B", "SCIPY"}:
            self.optimizer = None  # SciPy path uses a separate fitter
        else:
            raise NotImplementedError(f"Only ADAM/RMSPROP/LBFGSB supported here, got {optim}.")
        return self.optimizer


    def _loss_value(self) -> float: 
        self.gp_model.train()
        with torch.no_grad(): 
            out = self.gp_model(self.X)
            loss = -self.mll(out, self.y)
        return loss.item()

    @torch.enable_grad()
    def step(self, epochs: int, log_every: int = 0, verbose: bool = True,
            return_history: bool = False):
        self.gp_model.train(); self.likelihood.train()

        if getattr(self, "_optim_name", None) in {"LBFGSB", "LBFGS-B", "SCIPY"}:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model).to(self.device)
            try:
                from botorch.optim.fit import fit_gpytorch_mll_scipy
            except ImportError:
                from botorch.fit import fit_gpytorch_mll_scipy
            init_loss = self._loss_value()
            fit_gpytorch_mll_scipy(mll, options={"maxiter": int(epochs), "disp": False})
            end_loss = self._loss_value()
            if verbose:
                print(f"[GP_Wrapper.step][LBFGSB] {epochs} iters: {init_loss:.6f} → {end_loss:.6f} "
                    f"(Δ={init_loss - end_loss:.6f}, rel={(init_loss - end_loss)/abs(init_loss):.2%})")
            return (end_loss, []) if return_history else end_loss

        # fallback: PyTorch loop for ADAM/RMSPROP (unchanged)
        init_loss = 1e10
        losses = []
        for t in range(epochs):
            self.optimizer.zero_grad()
            out = self.gp_model(self.X)
            loss = -self.mll(out, self.y)
            loss.backward()
            self.optimizer.step()
            cur = loss.item()
            if t == 0: init_loss = cur
            if return_history: losses.append(cur)
            if log_every and (t + 1) % log_every == 0 and t > 0:
                print(f"[epoch {t+1:4d}] loss={cur:.6f} (Δ={init_loss - cur:.6f}, rel={(init_loss - cur)/abs(init_loss):.2%})")
        end_loss = self._loss_value()
        if verbose:
            print(f"[GP_Wrapper.step] {epochs} epochs: {init_loss:.6f} → {end_loss:.6f} "
                f"(Δ={init_loss - end_loss:.6f}, rel={(init_loss - end_loss)/abs(init_loss):.2%})")
        return (end_loss, losses) if return_history else end_loss


    # ---- NEW ----
    def update_train_data(self, train_x, train_y):
        self.X = train_x.to(self.device)
        self.y = train_y.squeeze().to(self.device)
        # crucial bit: do NOT recreate the model, just point it to the new data
        self.gp_model.set_train_data(inputs=self.X, targets=self.y, strict=False)


    def pred(self, test_x, num_samples=8):
        self.gp_model.eval()
        with torch.no_grad():
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