import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


Tensor = torch.Tensor


@dataclass
class BoundResult:
    lower: Tensor
    upper: Tensor
    A_dict_lower: Dict[int, Tensor]
    A_dict_upper: Dict[int, Tensor]
    alpha_dict: Dict[int, Tensor]
    preact_lbs: List[Tensor]
    preact_ubs: List[Tensor]


class AlphaCROWNSequential:
    """
    Minimal alpha-CROWN for Sequential(Linear, ReLU, ..., Linear).

    Conventions:
    - batch-major tensors
    - specification matrix C has shape (B, M, out_dim)
      where B=batch size, M=number of specs per example
    - all A tensors have shape (B, M, dim_of_that_layer)

    This implementation:
    - computes initial pre-activation bounds with IBP
    - optimizes alpha for unstable ReLUs when computing LOWER bounds
    - uses standard CROWN backward propagation
    - also returns upper bounds using fixed upper relaxations
    """

    def __init__(self, model: nn.Sequential, device: Optional[torch.device] = None):
        self.model = copy.deepcopy(model)
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.linears, self.relu_positions = self._parse_model()

    def _parse_model(self) -> Tuple[List[nn.Linear], List[int]]:
        linears: List[nn.Linear] = []
        relu_positions: List[int] = []

        modules = list(self.model)
        for idx, m in enumerate(modules):
            if isinstance(m, nn.Linear):
                linears.append(m)
            elif isinstance(m, nn.ReLU):
                relu_positions.append(idx)
            else:
                raise ValueError(
                    f"Unsupported module {type(m)}. "
                    "This minimal implementation only supports nn.Linear and nn.ReLU."
                )
        return linears, relu_positions

    @staticmethod
    def _split_pos_neg(x: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.clamp(x, min=0.0), torch.clamp(x, max=0.0)

    def ibp(self, x_L: Tensor, x_U: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Compute pre-activation bounds for every Linear layer.
        Returns:
            preact_lbs[k], preact_ubs[k] for k-th Linear layer output
        """
        lbs: List[Tensor] = []
        ubs: List[Tensor] = []

        l = x_L
        u = x_U

        for module in self.model:
            if isinstance(module, nn.Linear):
                W = module.weight  # (out, in)
                b = module.bias    # (out,)

                W_pos = torch.clamp(W, min=0.0)
                W_neg = torch.clamp(W, max=0.0)

                l_new = l @ W_pos.t() + u @ W_neg.t() + b
                u_new = u @ W_pos.t() + l @ W_neg.t() + b

                lbs.append(l_new)
                ubs.append(u_new)

                l, u = l_new, u_new

            elif isinstance(module, nn.ReLU):
                l = torch.clamp(l, min=0.0)
                u = torch.clamp(u, min=0.0)

            else:
                raise ValueError(f"Unsupported module {type(module)}.")

        return lbs, ubs

    def _init_alpha_dict(
        self,
        preact_lbs: List[Tensor],
        preact_ubs: List[Tensor],
    ) -> Dict[int, nn.Parameter]:
        """
        One alpha tensor per ReLU layer, parameterized at the dimension of that ReLU's input.
        alpha shape: (B, dim)
        Only meaningful on unstable neurons.
        """
        alpha_dict: Dict[int, nn.Parameter] = {}

        linear_idx = 0
        modules = list(self.model)
        for module in modules:
            if isinstance(module, nn.Linear):
                linear_idx += 1
            elif isinstance(module, nn.ReLU):
                lb = preact_lbs[linear_idx - 1]
                ub = preact_ubs[linear_idx - 1]
                # initialize alpha to secant-like midpoint 0.5 on unstable units
                init = torch.full_like(lb, 0.5, device=lb.device)
                alpha_dict[linear_idx - 1] = nn.Parameter(init)
        return alpha_dict

    @staticmethod
    def _relu_upper_params(lb: Tensor, ub: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Upper relaxation of ReLU:
            y <= s_u * x + b_u
        For stable cases:
            active: y = x
            inactive: y = 0
        """
        eps = 1e-12
        s_u = torch.zeros_like(lb)
        b_u = torch.zeros_like(lb)

        stable_active = lb >= 0
        stable_inactive = ub <= 0
        unstable = ~(stable_active | stable_inactive)

        s_u[stable_active] = 1.0
        b_u[stable_active] = 0.0

        s_u[stable_inactive] = 0.0
        b_u[stable_inactive] = 0.0

        s_u[unstable] = ub[unstable] / (ub[unstable] - lb[unstable] + eps)
        b_u[unstable] = -lb[unstable] * s_u[unstable]

        return s_u, b_u

    @staticmethod
    def _relu_lower_params_alpha(
        lb: Tensor,
        ub: Tensor,
        alpha_raw: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Lower relaxation of ReLU:
            y >= s_l * x + b_l
        This minimal alpha-CROWN uses:
            stable active: y = x
            stable inactive: y = 0
            unstable: y >= alpha * x, alpha in [0, 1]
        """
        s_l = torch.zeros_like(lb)
        b_l = torch.zeros_like(lb)

        stable_active = lb >= 0
        stable_inactive = ub <= 0
        unstable = ~(stable_active | stable_inactive)

        s_l[stable_active] = 1.0
        b_l[stable_active] = 0.0

        s_l[stable_inactive] = 0.0
        b_l[stable_inactive] = 0.0

        alpha = torch.clamp(alpha_raw, 0.0, 1.0)
        s_l[unstable] = alpha[unstable]
        b_l[unstable] = 0.0

        return s_l, b_l

    def _backward_bound(
        self,
        x_L: Tensor,
        x_U: Tensor,
        C: Tensor,
        preact_lbs: List[Tensor],
        preact_ubs: List[Tensor],
        alpha_dict: Dict[int, Tensor],
        compute_upper: bool = False,
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        """
        Backward CROWN / alpha-CROWN.

        Args:
            C: (B, M, out_dim)
            compute_upper:
                False => lower bound on C f(x)
                True  => upper bound on C f(x)

        Returns:
            bound: (B, M)
            A_dict: layer-indexed A tensors
        """
        modules = list(self.model)
        B = x_L.shape[0]

        # Start from output spec.
        A = C
        bias_term = torch.zeros(C.shape[0], C.shape[1], device=C.device, dtype=C.dtype)

        A_dict: Dict[int, Tensor] = {}

        # linear_idx counts outputs of Linear layers
        linear_idx = sum(isinstance(m, nn.Linear) for m in modules) - 1

        for module in reversed(modules):
            if isinstance(module, nn.Linear):
                # Save A associated with this linear output before propagating through the layer.
                A_dict[linear_idx] = A

                W = module.weight  # (out, in)
                b = module.bias    # (out,)

                # Accumulate bias contribution: A @ b
                bias_term = bias_term + torch.einsum("bmo,o->bm", A, b)

                # Backprop through affine: A <- A W
                A = torch.einsum("bmo,oi->bmi", A, W)

                linear_idx -= 1

            elif isinstance(module, nn.ReLU):
                # This ReLU sits after Linear layer with index linear_idx
                lb = preact_lbs[linear_idx]
                ub = preact_ubs[linear_idx]

                s_u, b_u = self._relu_upper_params(lb, ub)
                s_l, b_l = self._relu_lower_params_alpha(lb, ub, alpha_dict[linear_idx])

                A_pos, A_neg = self._split_pos_neg(A)

                if not compute_upper:
                    # Lower bound propagation:
                    # positive coeff => lower relaxation
                    # negative coeff => upper relaxation
                    slope = A_pos * s_l.unsqueeze(1) + A_neg * s_u.unsqueeze(1)
                    intercept = (
                        A_pos * b_l.unsqueeze(1) + A_neg * b_u.unsqueeze(1)
                    ).sum(dim=-1)
                else:
                    # Upper bound propagation:
                    # positive coeff => upper relaxation
                    # negative coeff => lower relaxation
                    slope = A_pos * s_u.unsqueeze(1) + A_neg * s_l.unsqueeze(1)
                    intercept = (
                        A_pos * b_u.unsqueeze(1) + A_neg * b_l.unsqueeze(1)
                    ).sum(dim=-1)

                A = slope
                bias_term = bias_term + intercept

            else:
                raise ValueError(f"Unsupported module {type(module)}.")

        # Final input-box evaluation
        A_pos, A_neg = self._split_pos_neg(A)

        if not compute_upper:
            bound = (
                (A_pos * x_L.unsqueeze(1)).sum(dim=-1)
                + (A_neg * x_U.unsqueeze(1)).sum(dim=-1)
                + bias_term
            )
        else:
            bound = (
                (A_pos * x_U.unsqueeze(1)).sum(dim=-1)
                + (A_neg * x_L.unsqueeze(1)).sum(dim=-1)
                + bias_term
            )

        A_dict[-1] = A  # input coefficients
        return bound, A_dict

    def compute_bounds(
        self,
        x_center: Tensor,
        eps: float,
        C: Tensor,
        alpha_steps: int = 20,
        alpha_lr: float = 1e-1,
        optimize_alpha: bool = True,
        verbose: bool = False,
    ) -> BoundResult:
        """
        Compute alpha-CROWN bounds for input box [x_center-eps, x_center+eps].

        Args:
            x_center: (B, in_dim)
            eps: scalar Linf radius
            C: (B, M, out_dim)
            alpha_steps: optimization steps for alpha
            alpha_lr: Adam learning rate
            optimize_alpha: whether to optimize alpha
        """
        x_center = x_center.to(self.device)
        C = C.to(self.device)

        x_L = x_center - eps
        x_U = x_center + eps

        with torch.no_grad():
            preact_lbs, preact_ubs = self.ibp(x_L, x_U)
        preact_lbs = [t.detach() for t in preact_lbs]
        preact_ubs = [t.detach() for t in preact_ubs]
        alpha_params = self._init_alpha_dict(preact_lbs, preact_ubs)

        if optimize_alpha and len(alpha_params) > 0:
            optimizer = torch.optim.Adam(alpha_params.values(), lr=alpha_lr)

            for step in range(alpha_steps):
                optimizer.zero_grad()

                lower, _ = self._backward_bound(
                    x_L=x_L,
                    x_U=x_U,
                    C=C,
                    preact_lbs=preact_lbs,
                    preact_ubs=preact_ubs,
                    alpha_dict=alpha_params,
                    compute_upper=False,
                )

                # maximize lower bound
                loss = -lower.mean()
                loss.backward()
                optimizer.step()

                # hard clamp after update
                for p in alpha_params.values():
                    p.data.clamp_(0.0, 1.0)

                if verbose and (step == 0 or step == alpha_steps - 1 or (step + 1) % 10 == 0):
                    print(f"[alpha step {step+1:03d}] mean lower = {lower.mean().item():.6f}")

        lower, A_dict_lower = self._backward_bound(
            x_L=x_L,
            x_U=x_U,
            C=C,
            preact_lbs=preact_lbs,
            preact_ubs=preact_ubs,
            alpha_dict=alpha_params,
            compute_upper=False,
        )

        upper, A_dict_upper = self._backward_bound(
            x_L=x_L,
            x_U=x_U,
            C=C,
            preact_lbs=preact_lbs,
            preact_ubs=preact_ubs,
            alpha_dict=alpha_params,
            compute_upper=True,
        )

        alpha_out = {k: v.detach().clone() for k, v in alpha_params.items()}

        return BoundResult(
            lower=lower.detach(),
            upper=upper.detach(),
            A_dict_lower={k: v.detach().clone() for k, v in A_dict_lower.items()},
            A_dict_upper={k: v.detach().clone() for k, v in A_dict_upper.items()},
            alpha_dict=alpha_out,
            preact_lbs=[t.detach().clone() for t in preact_lbs],
            preact_ubs=[t.detach().clone() for t in preact_ubs],
        )