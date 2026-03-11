import torch
import torch.nn as nn
import torch.nn.functional as F
from common.network import LayerType


# def select_layer(layer, pre_conv=None, post_conv=None, pre_lb=None, pre_ub=None):
#     if layer.type is LayerType.Linear:
#         return DualLinear(layer)
#     elif layer.type is LayerType.Conv2D:
#         if pre_conv is None or post_conv is None:
#             raise ValueError("pre_conv and post_conv must be provided for DualConv2D")
#         return DualConv2D(layer, pre_shape=pre_conv, post_shape=post_conv)
#     elif layer.type is LayerType.ReLU:
#         if pre_lb is None or pre_ub is None:
#             raise ValueError("pre_lb and pre_ub must be provided for DualRelu")
#         return DualRelu(pre_lb, pre_ub)
#     else:
#         raise ValueError(f"Unsupported layer type: {layer.type}")


def batch(A, n):
    return A.view(n, -1, *A.size()[1:])


def unbatch(A):
    return A.view(-1, *A.size()[2:])


class DualLinear_Ind():
    def __init__(self, layer):
        self.weight = layer.weight
        self.bias = layer.bias

    def T(self, inp_vs):
        inp_v = inp_vs[-1]
        if inp_v is None:
            return None
        new_inp_v = F.linear(inp_v, self.weight.t())  # Wt*inp_v
        # inp_v shape: (batch_size(n), out_dim) -> (c0, c1, c2, ..., cn)
        # weight shape: (out_dim, in_dim)
        # new_inp_v shape: (batch_size(n), in_dim) -> (c'0, c'1, c'2, ..., c'n)

        return new_inp_v

    def objective(self, A):
        if self.bias is None:
            return torch.tensor(0.0, device=A.device)
        
        bias_flat = self.bias.view(-1)  # (out_dim)

        if A.dim() == 1:
            return -A.matmul(bias_flat)
        elif A.dim() == 2:
            return -(A * bias_flat.unsqueeze(0)).sum(dim=1)  # batch-wise objective (batch_size(n))
        else:
            raise ValueError(f"Expected A to have 1 or 2 dimensions, got {A.dim()}")


class DualConv2D_Ind():
    def __init__(self, layer, in_shape, out_shape):
        self.in_shape = in_shape  # forward pass
        self.out_shape = out_shape  # forward pass
        self.weight = layer.weight
        self.stride = layer.stride
        self.padding = layer.padding
        self.group = 1
        self.dilation = layer.dilation
        # self.bias = layer.bias

        C_out, H_out, W_out = self.out_shape
        b = layer.bias.view(C_out, 1, 1).expand(C_out, H_out, W_out)
        self.bias = b.flatten()

    def conv_transpose2d(self, v):
        i = 0
        out = []
        batch_size = 10000

        kH, kW = self.weight.shape[-2:]
        C_out, out_H, out_W = self.out_shape
        C_in,  in_H,  in_W = self.in_shape

        # Compute output_padding so that transposed conv maps (out_H, out_W) -> (in_H, in_W)
        base_H = (out_H - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (kH - 1) + 1
        base_W = (out_W - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (kW - 1) + 1
        op_h = in_H - base_H
        op_w = in_W - base_W

        if not (0 <= op_h < self.stride[0] and 0 <= op_w < self.stride[1]):
            raise ValueError(f"Computed output_padding {(op_h, op_w)} invalid for stride {self.stride}. "
                             f"Check in/out shapes or conv params.")
        out_padding = (int(op_h), int(op_w))

        while i < v.size(0):
            out.append(F.conv_transpose2d(v[i:min(i+batch_size, v.size(0))], self.weight, None,
                                          self.stride, self.padding, out_padding, self.group, self.dilation))
            i += batch_size
        return torch.cat(out, 0)

    def conv_reshape(self, v):
        """
        1D (C*H*W) -> (1, C, H, W)
        2D (batch, C*H*W) -> (batch, C, H, W)
        """

        C, H, W = self.out_shape
        total_size = C * H * W

        if v.dim() == 1:
            if v.numel() != total_size:
                raise ValueError(f"Expected {total_size} elements, got {v.numel()}")
            return v.view(1, C, H, W), False
        elif v.dim() == 2:
            if v.size(1) != total_size:
                raise ValueError(f"Expected {total_size} elements per batch, got {v.size(1)}")
            return v.view(v.size(0), C, H, W), True
        else:
            raise ValueError(f"Expected v to have 1 or 2 dimensions, got {v.dim()}")

    def T(self, inp_vs):
        inp_v = inp_vs[-1]
        if inp_v is None:
            return None
        inp_v_4d, batched = self.conv_reshape(inp_v)
        new_inp_v_4d = self.conv_transpose2d(inp_v_4d)
        flat = new_inp_v_4d.view(new_inp_v_4d.size(0), -1)  # (batch_size(n), C*H*W)

        if batched:
            return flat
        else:
            return flat.view(-1)  # back to 1D (C*H*W)

    def objective(self, A):
        if self.bias is None:
            return torch.tensor(0.0, device=A.device)
        
        bias_flat = self.bias.view(-1)  # (out_dim)

        if A.dim() == 1:
            return -A.matmul(self.bias)
        elif A.dim() == 2:
            return -(A * bias_flat.unsqueeze(0)).sum(dim=1)  # batch-wise objective (batch_size(n))
        else:
            raise ValueError(f"Expected A to have 1 or 2 dimensions, got {A.dim()}")


class DualRelu_Ind():
    def __init__(self, inp_lb, inp_ub, relu_precise=False):
        self.inp_lb = inp_lb
        self.inp_ub = inp_ub
        self.positive_mask = (inp_lb >= 0) & (inp_ub > 0)
        self.negative_mask = (inp_ub <= 0)
        self.unstable_mask = (~self.positive_mask) & (~self.negative_mask)
        self.relu_precise = relu_precise

    def get_lambda(self, inp_v):
        inp_lb = self.inp_lb
        inp_ub = self.inp_ub

        # preactivation state is negative
        base_lam = torch.zeros_like(inp_lb)

        # preactivation state is positive
        base_lam = torch.where(self.positive_mask, torch.ones_like(inp_lb), base_lam)

        # preactivation state is unstable
        temp_c = inp_ub / (inp_ub - inp_lb + 1e-15)
        if not self.relu_precise:
            base_lam = torch.where(self.unstable_mask, temp_c, base_lam)
            if inp_v.dim() == 1:
                # 1D case
                return base_lam
            elif inp_v.dim() == 2:
                # batched case
                batch_size = inp_v.size(0)
                return base_lam.unsqueeze(0).expand(batch_size, -1)
            else:
                raise ValueError(f"Expected inp_v to have 1 or 2 dimensions, got {inp_v.dim()}")
        else:
            unstable = self.unstable_mask
            slope_zero = (-inp_lb >= inp_ub) & unstable  # lower bound is y=0
            slope_one = (inp_ub > -inp_lb) & unstable    # lower bound is y=x
            if inp_v.dim() == 1:
                # 1D case
                lam = base_lam.clone()
                lam = torch.where(unstable & (inp_v >= 0), temp_c, lam)
                lam = torch.where(slope_one & (inp_v < 0), torch.ones_like(inp_lb), lam)
                return lam
            elif inp_v.dim() == 2:
                # batched case
                batch_size = inp_v.size(0)
                lam = base_lam.unsqueeze(0).expand(batch_size, -1).clone()
                
                # (dim) -> (1, dim)
                unstable_b = unstable.unsqueeze(0)
                slope_zero_b = slope_zero.unsqueeze(0)
                slope_one_b = slope_one.unsqueeze(0)
                temp_c_b = temp_c.unsqueeze(0)

                pos = (inp_v >= 0)  # (batch_size, dim)
                neg = (inp_v < 0)

                lam = torch.where(unstable_b & pos, temp_c_b, lam)
                lam = torch.where(slope_one_b & neg, torch.ones_like(lam), lam)

                return lam
            else:
                raise ValueError(f"Expected inp_v to have 1 or 2 dimensions, got {inp_v.dim()}")

    def T(self, inp_vs):
        inp_v = inp_vs[-1]
        if inp_v is None:
            return None

        lam = self.get_lambda(inp_v)

        new_inp_v = lam * inp_v
        return new_inp_v

    def objective(self, A):
        temp_coef = (self.inp_ub*self.inp_lb)/(self.inp_ub - self.inp_lb + 1e-15)
        coef = torch.where(self.unstable_mask, temp_coef, torch.zeros_like(temp_coef))

        if A.dim() == 1:
            pos_A = torch.clamp(A, min=0)
            return pos_A.matmul(coef)
        elif A.dim() == 2:
            # batched A (batch_size(n), dim)
            pos_A = torch.clamp(A, min=0)
            return (pos_A * coef).sum(dim=1)  # batch-wise objective (batch_size(n))
        else:
            raise ValueError(f"Expected A to have 1 or 2 dimensions, got {A.dim()}")
