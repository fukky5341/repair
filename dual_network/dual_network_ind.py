import torch
import torch.nn as nn
import torch.nn.functional as F
from dual_network.dual_layers_ind import (
    DualLinear_Ind, DualConv2D_Ind, DualRelu_Ind, DualFlatten_Ind
)


class DualNetwork_Ind:
    """
    Dual network for individual specification.

    C:
        shape (out_dim,) or (batch_specs, out_dim)

    This version supports:
    - nn.Sequential
    - alpha re-optimization inside the dual network
    """

    def __init__(self, C, ori_net, shapes, lbs, ubs, relu_precise=False, alpha_params=None):
        self.dual_net = []
        self.As = []  # As[::-1]: [A(L0), A(R1), A(L2), A(R3), ...]

        if C.dim() == 1:
            C = C.unsqueeze(0)

        self.C = C.detach() if C.requires_grad else C
        self.batch_size = C.size(0)
        self.ori_net = ori_net
        self.shapes = shapes

        # IMPORTANT: treat neuron bounds as constants during dual alpha optimization
        self.lbs = [x.detach() for x in lbs]
        self.ubs = [x.detach() for x in ubs]

        self.relu_precise = relu_precise
        self.alpha_params = {} if alpha_params is None else alpha_params

    def build_dual_network_individual(self):
        dual_net = []

        for layer_idx, layer in enumerate(self.ori_net):
            if isinstance(layer, nn.Linear):
                dual_layer = DualLinear_Ind(layer)

            elif isinstance(layer, nn.Conv2d):
                affine_layer_index = sum(
                    isinstance(self.ori_net[i], (nn.Linear, nn.Conv2d))
                    for i in range(layer_idx + 1)
                ) - 1

                dual_layer = DualConv2D_Ind(
                    layer,
                    in_shape=self.shapes[affine_layer_index],
                    out_shape=self.shapes[affine_layer_index + 1],
                )

            elif isinstance(layer, nn.ReLU):
                alpha = self.alpha_params.get(layer_idx, None)
                dual_layer = DualRelu_Ind(
                    self.lbs[layer_idx],
                    self.ubs[layer_idx],
                    relu_precise=self.relu_precise,
                    alpha=alpha,
                )

            elif isinstance(layer, nn.Flatten):
                dual_layer = DualFlatten_Ind()

            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")

            dual_net.append(dual_layer)

        self.dual_net = dual_net
        self.rebuild_As()
        return

    def rebuild_As(self):
        """
        Recompute backward coefficients As using current alpha values.
        """
        As = [-self.C]
        for r_layer in reversed(self.dual_net):
            curr_A = r_layer.T(As)
            As.append(curr_A)
        self.As = As

    def squeeze_As_to_1d(self):
        def _maybe_squeeze(A):
            if A is None:
                return None
            if A.dim() == 2 and A.size(0) == 1:
                return A.squeeze(0)
            return A

        self.As = [_maybe_squeeze(A) for A in self.As]

    def get_minimized_objective(self):
        """
        Returns minimized dual objective for the current C and current alpha.
        """
        if len(self.dual_net) + 1 != len(self.As):
            raise ValueError("Length mismatch between dual_net and As")

        input_obj = self.input_objective(self.As[-1])

        reversed_As = list(reversed(self.As))
        total_objective = input_obj

        for i, layer in enumerate(self.dual_net):
            obj = layer.objective(reversed_As[i + 1])
            total_objective = total_objective + obj

        if total_objective.dim() == 0:
            return total_objective
        elif total_objective.dim() == 1 and total_objective.size(0) == 1:
            return total_objective[0]
        else:
            return total_objective

    def input_objective(self, A):
        if A is None:
            return torch.tensor(0.0, device=self.C.device, dtype=self.C.dtype)

        inp_lb = self.lbs[0]
        inp_ub = self.ubs[0]
        lb_flat = inp_lb.view(-1)
        ub_flat = inp_ub.view(-1)

        if A.dim() == 1:
            A_pos = torch.clamp(A, min=0)
            A_neg_abs = -torch.clamp(A, max=0)
            obj = -A_pos.matmul(ub_flat) + A_neg_abs.matmul(lb_flat)
            return obj

        elif A.dim() == 2:
            A_pos = torch.clamp(A, min=0)
            A_neg_abs = -torch.clamp(A, max=0)
            obj = -(A_pos * ub_flat.unsqueeze(0)).sum(dim=1) + (A_neg_abs * lb_flat.unsqueeze(0)).sum(dim=1)
            return obj

        else:
            raise ValueError(f"Expected A to have 1 or 2 dims, got {A.dim()}")

    # ------------------------------------------------------------
    # alpha optimization helpers
    # ------------------------------------------------------------

    def get_alpha_parameters(self):
        """
        Collect alpha tensors from DualRelu_Ind layers.
        """
        params = []
        for layer in self.dual_net:
            if isinstance(layer, DualRelu_Ind) and layer.alpha is not None:
                params.append(layer.alpha)
        return params

    def clamp_alpha(self):
        for layer in self.dual_net:
            if isinstance(layer, DualRelu_Ind) and layer.alpha is not None:
                layer.alpha.data.clamp_(0.0, 1.0)

    def optimize_alpha(self, steps=30, lr=1e-2, verbose=False):
        """
        Optimize alpha for the current specification C.

        Solves approximately:
            max_alpha g(alpha; C)
        """
        alpha_params = self.get_alpha_parameters()
        if len(alpha_params) == 0:
            if verbose:
                print("No alpha parameters found in dual network.")
            return

        optimizer = torch.optim.Adam(alpha_params, lr=lr)

        for step in range(steps):
            optimizer.zero_grad()

            # Recompute backward coefficients using current alpha.
            self.rebuild_As()

            obj = self.get_minimized_objective()

            # maximize objective
            loss = -obj.mean()
            loss.backward()
            optimizer.step()

            self.clamp_alpha()

            if verbose and (step == 0 or step == steps - 1 or (step + 1) % 10 == 0):
                print(f"[dual alpha step {step+1:03d}] objective = {obj.detach()}")

        # final rebuild after optimization
        self.rebuild_As()

    # ------------------------------------------------------------
    # Sliced dual network
    # ------------------------------------------------------------
    """
    Assumption:
        affine and relu layers alternate (e.g., Linear(0) -> ReLU(1) -> Linear(2) -> ReLU(3) -> ...)

    Given layer index of Linear layer to repair is Li:
        - As[::-1]: [A(L0), A(R1), ..., A(Li), A(Ri+1), A(Li+2), A(Ri+3), ...]
        - necessary As: As[::-1][Li+2:] = [A(Li+2), A(Ri+3), ...]
        - dual layers: [L0, R1, ..., Li, Ri+1, Li+2, Ri+3, ...]
        - necessary dual layers: dual_net[Li+2:] = [Li+2, Ri+3, ...]
    
    Objective for sliced dual network (DualNet[Li+2:]):
        obj = sum(constant term from A(Ri+3), A(Li+4), ...) 
                + (A(Li+2) term that depends on the ouput bounds N[:Li+2] = N[L0, ..., Li, Ri+1]) <-- like input obj
    """
    def sliced_As(self, slice_layer_idx):
        sliced_As = self.As[::-1][slice_layer_idx:]
        return sliced_As[::-1]  # reverse back to match dual_net order
    
    def sliced_dual_net(self, slice_layer_idx):
        sliced_dual_net = self.dual_net[slice_layer_idx:]
        return sliced_dual_net
    
    def sliced_input_objective(self, As0, lb, ub):
        '''
        As0: 1D
        lb, ub: 1D
        '''
        As0_pos  = torch.clamp(As0, min=0)
        A_neg_abs = -torch.clamp(As0, max=0)
        obj = -As0_pos.matmul(ub) + A_neg_abs.matmul(lb)
        return obj
    
    def sliced_subseq_layer_objective(self, sliced_dual_net, sliced_As):
        total_obj = 0
        reversed_sliced_As = sliced_As[::-1]
        for i, layer in enumerate(sliced_dual_net):
            obj = layer.objective(reversed_sliced_As[i + 1])
            total_obj = total_obj + obj
        return total_obj
    
    def sliced_objective(self, repaired_layer_idx, lb, ub, subseq_layers_obj_keep=None):
        ''' 
        repaired_layer_idx: index of the repaired affine layer (Li)
        lb, ub: 
            - neuron bounds for the output of the repaired layer (N[:Li+2] = N[..., Li, Ri+1])
            - 1D shape (no batch)
        As:
            - 1D shape (no batch)
        
        ------------------------------------------------------------
        returns:
            obj = sum(constant term from A(Ri+3), A(Li+4), ...) + (A(Li+2) term that depends on the ouput bounds N[:Li+2])
        '''
        assert As0.dim() == 1, "sliced_objective currently supports only non-batched As"
        assert lb.dim() == 1 and ub.dim() == 1
        
        # As
        sliced_As = self.sliced_As(repaired_layer_idx + 2)
        # dual net
        sliced_dual_net = self.sliced_dual_net(repaired_layer_idx + 2)

        # --- compute objective ---
        # sum(constant term from Ri+3, Li+4, ...)
        if subseq_layers_obj_keep is not None:
            subseq_layers_obj = subseq_layers_obj_keep
        else:
            subseq_layers_obj = self.sliced_subseq_layer_objective(sliced_dual_net, sliced_As)
        # (Li+2 term that depends on the ouput bounds N[:Li+2])
        As0 = sliced_As[::-1][0]  # A(Li+2)
        sliced_input_obj = self.sliced_input_objective(As0, lb, ub)
        # total obj
        total_obj = subseq_layers_obj + sliced_input_obj

        return total_obj
