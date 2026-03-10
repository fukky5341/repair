import torch
import torch.nn.functional as F
import gc
from common.network import LayerType
from util.util import compute_input_shapes


class BackPropStruct:
    """
    Out = C@In + b

    Out, In <-- vector
    C <-- coefficient matrix
    b <-- bias vector
    """

    def __init__(self):
        self.n_C_lb = None
        self.n_b_lb = None
        self.n_C_ub = None
        self.n_b_ub = None

    def populate(self, n_C_lb, n_b_lb, n_C_ub, n_b_ub):
        self.n_C_lb = n_C_lb
        self.n_b_lb = n_b_lb
        self.n_C_ub = n_C_ub
        self.n_b_ub = n_b_ub

    def delete_old(self):
        del self.n_C_lb, self.n_b_lb, self.n_C_ub, self.n_b_ub
        gc.collect()
        torch.cuda.empty_cache()

    def print_Cb(self):
        print(f"n_C_lb: {self.n_C_lb}")
        print(f"n_C_ub: {self.n_C_ub}")
        print(f"n_b_lb: {self.n_b_lb}")
        print(f"n_b_ub: {self.n_b_ub}")


class IndividualBounds:
    def __init__(self, inp_prop, net, lb_inp, ub_inp, eps=None, device='') -> None:
        self.inp = inp_prop.input
        # if self.inp.shape[0] == 784:
        #     self.input_shape = (1, 28, 28)
        # elif self.inp.shape[0] == 3072:
        #     self.input_shape = (3, 32, 32)
        # elif self.inp.shape[0] == 2:
        #     self.input_shape = (1, 1, 2)
        # elif self.inp.shape[0] == 12:
        #     self.input_shape = (1, 1, 12)
        # elif self.inp.shape[0] == 87:
        #     self.input_shape = (1, 1, 87)
        # else:
        #     raise ValueError(f"Unrecognised input shape {self.input_shape}")
        self.input_shape = self.inp.shape
        if self.input_shape not in [(1, 28, 28), (3, 32, 32), (1, 1, 2), (1, 1, 12), (1, 1, 87), (2,), (5,)]:
            raise ValueError(f"Unrecognised input shape {self.input_shape}")
        self.net = net
        self.lb_inp = lb_inp
        self.ub_inp = ub_inp
        self.eps = eps
        self.shapes = compute_input_shapes(net=self.net, input_shape=self.input_shape)
        self.linear_conv_layer_indices = []
        self.device = device
        self.lbs = None
        self.ubs = None

    def get_layer_size(self, linear_layer_index):
        layer = self.net[self.linear_conv_layer_indices[linear_layer_index]]
        if layer.type is LayerType.Linear:
            shape = self.shapes[linear_layer_index + 1]
            return shape
        if layer.type is LayerType.Conv2D:
            shape = self.shapes[linear_layer_index + 1]
            return (shape[0] * shape[1] * shape[2])

    def initialize_back_prop_struct(self, layer_idx):

        layer_size = self.get_layer_size(linear_layer_index=layer_idx)
        n_C_lb = torch.eye(n=layer_size, device=self.device)
        n_b_lb = torch.zeros(layer_size, device=self.device)
        n_C_ub = torch.eye(n=layer_size, device=self.device)
        n_b_ub = torch.zeros(layer_size, device=self.device)

        back_prop_struct = BackPropStruct()
        back_prop_struct.populate(n_C_lb=n_C_lb,
                                  n_b_lb=n_b_lb,
                                  n_C_ub=n_C_ub,
                                  n_b_ub=n_b_ub)
        return back_prop_struct

    def pos_neg_weight_decomposition(self, coef):
        neg_comp = torch.where(coef < 0, coef, torch.zeros_like(coef, device=self.device))
        pos_comp = torch.where(coef >= 0, coef, torch.zeros_like(coef, device=self.device))
        return neg_comp, pos_comp

    def handle_linear_individual(self, linear_wt, bias, back_prop_struct):
        """
        Y = CX + b --> Y = C(WX' + b') + b = CWX' + b + Cb'
        """

        n_b_lb = back_prop_struct.n_b_lb + back_prop_struct.n_C_lb.matmul(bias)
        n_b_ub = back_prop_struct.n_b_ub + back_prop_struct.n_C_ub.matmul(bias)

        n_C_lb = back_prop_struct.n_C_lb.matmul(linear_wt)
        n_C_ub = back_prop_struct.n_C_ub.matmul(linear_wt)

        back_prop_struct.populate(n_b_lb=n_b_lb, n_b_ub=n_b_ub,
                                  n_C_lb=n_C_lb, n_C_ub=n_C_ub)

        return back_prop_struct

    def handle_conv_individual(self, conv_weight, conv_bias, back_prop_struct, preconv_shape, postconv_shape,
                               stride, padding, groups=1, dilation=(1, 1)):
        """
        Y = CX + b --> Y = C(WX' + b') + b = CWX' + b + Cb'
        """

        kernel_hw = conv_weight.shape[-2:]
        h_padding = (preconv_shape[1] + 2 * padding[0] - 1 - dilation[0] * (kernel_hw[0] - 1)) % stride[0]
        w_padding = (preconv_shape[2] + 2 * padding[1] - 1 - dilation[1] * (kernel_hw[1] - 1)) % stride[1]
        output_padding = (h_padding, w_padding)

        coef_shape = back_prop_struct.n_C_lb.shape
        n_C_lb = back_prop_struct.n_C_lb.view((coef_shape[0], *postconv_shape))
        n_C_ub = back_prop_struct.n_C_ub.view((coef_shape[0], *postconv_shape))

        n_b_lb = back_prop_struct.n_b_lb + (n_C_lb.sum((2, 3)) * conv_bias).sum(1)
        n_b_ub = back_prop_struct.n_b_ub + (n_C_ub.sum((2, 3)) * conv_bias).sum(1)

        n_C_lb = F.conv_transpose2d(n_C_lb, conv_weight, None, stride, padding,
                                    output_padding, groups, dilation)
        n_C_ub = F.conv_transpose2d(n_C_ub, conv_weight, None, stride, padding,
                                    output_padding, groups, dilation)

        n_C_lb = n_C_lb.view((coef_shape[0], -1))
        n_C_ub = n_C_ub.view((coef_shape[0], -1))

        back_prop_struct.populate(
            n_C_lb=n_C_lb, n_C_ub=n_C_ub,
            n_b_lb=n_b_lb, n_b_ub=n_b_ub)

        return back_prop_struct

    def handle_relu_individual(self, back_prop_struct,
                               n_lb_layer, n_ub_layer):
        """
        Y = CX + b --> Y = C(LX' + m) + b = CLX' + b +Cm

        * y = ReLU(x) *
        (active) x <= y <= x
        (inactive) 0 <= y <= 0
        (unsettled) (0 or x) <= y <= u(x-l)/(u-l) --> y = lx + m
        """

        # $ --> debug
        # print(f"n_lb_layer: {n_lb_layer}")
        # print(f"n_ub_layer: {n_ub_layer}")
        # $ <-- debug

        # condition of relu
        relu_active = (n_lb_layer >= 0)
        relu_passive = (n_ub_layer <= 0)
        relu_unsettled = ~(relu_active) & ~(relu_passive)

        lambda_lb = torch.zeros(n_lb_layer.size(), device=self.device)
        lambda_ub = torch.zeros(n_lb_layer.size(), device=self.device)
        mu_ub = torch.zeros(n_lb_layer.size(), device=self.device)

        # active y = x
        lambda_lb = torch.where(relu_active, torch.ones(n_lb_layer.size(), device=self.device), lambda_lb)
        lambda_ub = torch.where(relu_active, torch.ones(n_lb_layer.size(), device=self.device), lambda_ub)

        # unsettled y = lx + m
        temp = torch.where(n_ub_layer < -n_lb_layer, torch.zeros(n_lb_layer.size(), device=self.device), torch.ones(n_lb_layer.size(), device=self.device))
        lambda_lb = torch.where(relu_unsettled, temp, lambda_lb)
        lambda_ub = torch.where(relu_unsettled, n_ub_layer/(n_ub_layer - n_lb_layer + 1e-15), lambda_ub)
        mu_ub = torch.where(relu_unsettled, -(n_ub_layer * n_lb_layer) / (n_ub_layer - n_lb_layer + 1e-15), mu_ub)

        neg_C_lb, pos_C_lb = self.pos_neg_weight_decomposition(back_prop_struct.n_C_lb)
        neg_C_ub, pos_C_ub = self.pos_neg_weight_decomposition(back_prop_struct.n_C_ub)

        n_b_lb = back_prop_struct.n_b_lb + neg_C_lb @ mu_ub
        n_b_ub = back_prop_struct.n_b_ub + pos_C_ub @ mu_ub

        n_C_lb = neg_C_lb * lambda_ub + pos_C_lb * lambda_lb
        n_C_ub = neg_C_ub * lambda_lb + pos_C_ub * lambda_ub

        back_prop_struct.populate(n_C_lb=n_C_lb, n_C_ub=n_C_ub,
                                  n_b_lb=n_b_lb, n_b_ub=n_b_ub)

        return back_prop_struct

    def handle_layer_individual(self, back_prop_struct, layer, linear_layer_idx, layer_idx, n_lbs, n_ubs):

        if layer.type is LayerType.Linear:
            back_prop_struct = self.handle_linear_individual(linear_wt=layer.weight,
                                                             bias=layer.bias, back_prop_struct=back_prop_struct)
        elif layer.type is LayerType.Conv2D:
            back_prop_struct = self.handle_conv_individual(conv_weight=layer.weight, conv_bias=layer.bias,
                                                           back_prop_struct=back_prop_struct,
                                                           preconv_shape=self.shapes[linear_layer_idx], postconv_shape=self.shapes[linear_layer_idx + 1],
                                                           stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
        elif layer.type is LayerType.ReLU:
            back_prop_struct = self.handle_relu_individual(back_prop_struct=back_prop_struct,
                                                           n_lb_layer=n_lbs[layer_idx], n_ub_layer=n_ubs[layer_idx])
        else:
            raise NotImplementedError(f'diff verifier for {layer.type} is not implemented')
        return back_prop_struct

    def check_lb_ub_correctness(self, lb, ub):
        if not torch.all(lb <= ub + 1e-6):
            return True
        return False

    def concretize_bounds_individual(self, back_prop_struct, n_lb_layer, n_ub_layer, layer_idx=None):
        """
        Y = (W^+)X + (W^-)X +b
        W <-- obtained by Back Propagation (symbolic bounds)
        X <-- concrete bounds 
        """

        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(back_prop_struct.n_C_lb)
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(back_prop_struct.n_C_ub)

        lb = neg_comp_lb @ n_ub_layer + pos_comp_lb @ n_lb_layer + back_prop_struct.n_b_lb
        ub = neg_comp_ub @ n_lb_layer + pos_comp_ub @ n_ub_layer + back_prop_struct.n_b_ub
        # TODO: it is possible to refine by the method like refine_diff_bounds?

        self.check_lb_ub_correctness(lb=lb, ub=ub)
        return lb, ub

    def backsubstitution_individual(self, layer_idx, n_lbs, n_ubs):

        back_prop_struct = None
        n_lb = None
        n_ub = None

        # Assuming the network has alternate affine and activation layer.
        linear_layer_index = layer_idx // 2
        for r_layer_idx in reversed(range(layer_idx + 1)):

            if back_prop_struct is None:
                back_prop_struct = self.initialize_back_prop_struct(layer_idx=linear_layer_index)
                # print(f"initialized")

            curr_layer = self.net[r_layer_idx]
            # print(f"{curr_layer.type}")
            back_prop_struct = self.handle_layer_individual(back_prop_struct=back_prop_struct, layer=curr_layer,
                                                            linear_layer_idx=linear_layer_index,
                                                            layer_idx=r_layer_idx, n_lbs=n_lbs, n_ubs=n_ubs)
            # print(f"handled")

            if r_layer_idx == 0:
                new_n_lb, new_n_ub = self.concretize_bounds_individual(back_prop_struct=back_prop_struct, n_lb_layer=n_lbs[r_layer_idx], n_ub_layer=n_ubs[r_layer_idx])
                n_lb = (new_n_lb if n_lb is None else (torch.max(n_lb, new_n_lb)))
                n_ub = (new_n_ub if n_ub is None else (torch.min(n_ub, new_n_ub)))
                # print(f"concretized: {n_lb}, {n_ub}")
                # print(f"concretized at layer {r_layer_idx}")

            # print(f"r_layer_idx: {r_layer_idx}")  # $ debug
            # back_prop_struct.print_Cb()

            if curr_layer.type in [LayerType.Linear, LayerType.Conv2D]:
                linear_layer_index -= 1

        lbub_alarm = self.check_lb_ub_correctness(lb=new_n_lb, ub=new_n_ub)
        if lbub_alarm:  # true: inconsistent bounds
            return None, None

        n_lb = (new_n_lb if n_lb is None else (torch.max(n_lb, new_n_lb)))
        n_ub = (new_n_ub if n_ub is None else (torch.min(n_ub, new_n_ub)))

        return n_lb, n_ub

    def run_backsubstitution_individual(self):

        # n_lbs = []
        # n_ubs = []
        # $ change -->
        self.lbs = [self.lb_inp]
        self.ubs = [self.ub_inp]
        # $ <-- change

        for layer_idx, layer in enumerate(self.net):
            # #$ change -->
            # if layer_idx == 0:
            #     curr_n_lb = self.lb_inp
            #     curr_n_ub = self.ub_inp
            # else:
            #     curr_n_lb, curr_n_ub = self.back_substitution_individual(layer_idx=layer_idx,
            #                                                             n_lbs=n_lbs, n_ubs=n_ubs)
            # #$ <-- change
            curr_n_lb, curr_n_ub = self.backsubstitution_individual(layer_idx=layer_idx, n_lbs=self.lbs, n_ubs=self.ubs)  # $ original
            if curr_n_lb is None or curr_n_ub is None:
                return None, None

            self.lbs.append(curr_n_lb)
            self.ubs.append(curr_n_ub)
            # print("layer: ", layer_idx, "\n", n_lbs,  "\n", n_ubs, "\n")  # $ debug

        return self.lbs, self.ubs

    def update_bounds_individual(self, split_history=None):
        if split_history is None or len(split_history) == 0:
            return self
        split_layer_list = [split_info.layer_idx for split_info in split_history]
        split_layer_list = list(set(split_layer_list))  # unique layer indices, e.g., [1,1,3,3,5] -> [1,3,5]
        start_layer_idx = min(split_layer_list)
        # truncate the bounds lists
        self.lbs = self.lbs[:start_layer_idx + 1]  # e.g., if start_layer_idx = 2, keep lubs[0], lubs[1], lubs[2]
        self.ubs = self.ubs[:start_layer_idx + 1]
        for layer_idx in range(start_layer_idx, len(self.net)):
            if self.net[layer_idx].type == LayerType.ReLU and layer_idx in split_layer_list:  # update bounds of preactivation layer
                temp_split_info_list = [split_info for split_info in split_history if split_info.layer_idx == layer_idx]
                if temp_split_info_list is not None:
                    for split_info in temp_split_info_list:
                        inp_lb = self.lbs[layer_idx][split_info.pos]
                        inp_ub = self.ubs[layer_idx][split_info.pos]

                        # update bounds
                        if split_info.split_type in ('ISA1', 'ISB1'):
                            new_inp_lb = inp_lb  # lb = lb
                            new_inp_ub = 0  # ub = 0
                        elif split_info.split_type in ('ISA2', 'ISB2'):
                            new_inp_lb = 0  # lb = 0
                            new_inp_ub = inp_ub  # ub = ub
                        else:
                            raise ValueError(f"Unknown split type: {split_info.split_type}")

                        self.lbs[layer_idx][split_info.pos] = new_inp_lb
                        self.ubs[layer_idx][split_info.pos] = new_inp_ub

            curr_lb, curr_ub = self.backsubstitution_individual(layer_idx=layer_idx, n_lbs=self.lbs, n_ubs=self.ubs)
            if curr_lb is None or curr_ub is None:
                return None

            self.lbs.append(curr_lb)
            self.ubs.append(curr_ub)

        return self

    def run(self):
        with torch.no_grad():
            for ind, layer in enumerate(self.net):
                if layer.type in [LayerType.Linear, LayerType.Conv2D]:
                    self.linear_conv_layer_indices.append(ind)

            return self.run_backsubstitution_individual()
