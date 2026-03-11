import gc
import torch
import torch.nn as nn
import warnings
from common.network import LayerType
from dual.dual_layers_ind import DualLinear_Ind, DualConv2D_Ind, DualRelu_Ind
from dual.dual_analysis_ind import DualAnalysis_Ind


class DualNetwork_Ind():  # for individual
    def __init__(self, C, ori_net, shapes, lbs, ubs, relu_precise=False):
        self.dual_net = []
        self.As = []  # for individual
        if C.dim() == 1:
            C = C.unsqueeze(0) # -> (1, n) to handle batch size 1
        self.C = C
        self.batch_size = C.size(0)
        self.ori_net = ori_net
        self.shapes = shapes
        self.lbs = lbs
        self.ubs = ubs
        self.relu_precise = relu_precise
        self.dual_analysis = DualAnalysis_Ind
        self.build_dual_network = self.build_dual_network_individual

    def build_dual_network_individual(self):
        dual_net = []

        for layer_idx, layer in enumerate(self.ori_net):
            if layer.type is LayerType.Linear:
                dual_layer = DualLinear_Ind(layer)
            elif layer.type is LayerType.Conv2D:
                # Assuming the network has alternate affine and activation layers.
                # input_shapes includes the shapes of the linear layers.
                linear_layer_index = layer_idx // 2
                dual_layer = DualConv2D_Ind(layer, in_shape=self.shapes[linear_layer_index], out_shape=self.shapes[linear_layer_index + 1])
            elif layer.type is LayerType.ReLU:
                lubs_idx = layer_idx
                dual_layer = DualRelu_Ind(self.lbs[lubs_idx], self.ubs[lubs_idx], self.relu_precise)
            else:
                raise ValueError(f"Unsupported layer type: {layer.type}")

            dual_net.append(dual_layer)
        self.dual_net = dual_net

        As = [-self.C]
        # depends on the objective
        # shape of C: (batch_size(n), layer_output_dim)  -> (C0, C1, C2, ..., Cn)

        for r_layer in reversed(dual_net):
            curr_As = r_layer.T(As)
            As.append(curr_As)

        self.As = As

        return

    def squeeze_As_to_1d(self):
        if self.batched:
            raise RuntimeError("Cannot squeeze As to 1D: dual network is batched.")

        def _maybe_squeeze(A):
            if A is None:
                return None
            if A.dim() == 2 and A.size(0) == 1:
                return A.squeeze(0)   # (1, dim) -> (dim,)
            return A

        self.As = [ _maybe_squeeze(A) for A in self.As ]

    def get_minimized_objective(self):
        """
        this provides minimized objective value for given C

        dual_net: [dual_l1, dual_r1, dual_l2, dual_r2, dual_l3]
        As: [-C, A'3, A2, A'2, A1, A'1]
        mapping of objective:
            A'1 (reversed As[0]) -> input
            A1 (reversed As[1]) -> dual_l1
            A'2 (reversed As[2]) -> dual_r1
            A2 (reversed As[3]) -> dual_l2
            A'3 (reversed As[4]) -> dual_r2
            -C (reversed As[5]) -> dual_l3
        """
        # check lengths
        if len(self.dual_net) + 1 != len(self.As):
            raise ValueError("Length mismatch between dual_net and As")
        # input
        input_obj = self.input_objective(self.As[-1])

        # other layers
        reversed_As = list(reversed(self.As))
        total_objective = input_obj

        for i, layer in enumerate(self.dual_net):
            obj = layer.objective(reversed_As[i + 1])
            total_objective = total_objective + obj
        
        if total_objective.dim() == 0:
            return total_objective  # scalar tensor
        elif total_objective.dim() == 1 and total_objective.size(0) == 1:
            return total_objective[0]  # tensor with single value
        else:
            return total_objective  # (batch_size(n),)

    def input_objective(self, A):
        if A is None:
            return torch.tensor(0.0, device=A.device)
        
        # input lb and ub
        inp_lb = self.lbs[0]
        inp_ub = self.ubs[0]
        lb_flat = inp_lb.view(-1)
        ub_flat = inp_ub.view(-1)

        if A.dim() == 1:
            abs_A_pos = torch.clamp(A, min=0)
            abs_A_neg = -torch.clamp(A, max=0)
            obj = -abs_A_pos.matmul(ub_flat) + abs_A_neg.matmul(lb_flat)
            return obj
        elif A.dim() == 2:
            # batched A (batch_size(n), input_dim)
            abs_A_pos = torch.clamp(A, min=0)
            abs_A_neg = -torch.clamp(A, max=0)
            obj = -(abs_A_pos * ub_flat).sum(dim=1) + (abs_A_neg * lb_flat).sum(dim=1)
            return obj  # (batch_size(n),)
        else:
            raise ValueError(f"Expected A to have 1 or 2 dimensions, got {A.dim()}")
