import torch
import torch.nn as nn
import torch.nn.functional as F


def batch(A, n):
    return A.view(n, -1, *A.size()[1:])


def unbatch(A):
    return A.view(-1, *A.size()[2:])


class DualLinear_Ind:
    def __init__(self, layer):
        self.weight = layer.weight
        self.bias = layer.bias

    def T(self, inp_As):
        inp_A = inp_As[-1]
        if inp_A is None:
            return None

        # If forward is y = W x + b, then backward coefficient is A W
        # F.linear(inp_A, self.weight.t()) computes inp_A @ self.weight
        new_inp_A = F.linear(inp_A, self.weight.t())
        return new_inp_A

    def objective(self, A):
        if self.bias is None:
            return torch.tensor(0.0, device=A.device, dtype=A.dtype)

        bias_flat = self.bias.view(-1)

        if A.dim() == 1:
            return -A.matmul(bias_flat)
        elif A.dim() == 2:
            return -(A * bias_flat.unsqueeze(0)).sum(dim=1)
        else:
            raise ValueError(f"Expected A to have 1 or 2 dims, got {A.dim()}")


class DualConv2D_Ind:
    def __init__(self, layer, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.weight = layer.weight
        self.stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
        self.padding = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
        self.group = layer.groups
        self.dilation = layer.dilation if isinstance(layer.dilation, tuple) else (layer.dilation, layer.dilation)

        C_out, H_out, W_out = self.out_shape
        if layer.bias is None:
            self.bias = None
        else:
            b = layer.bias.view(C_out, 1, 1).expand(C_out, H_out, W_out)
            self.bias = b.flatten()

    def conv_transpose2d(self, v):
        i = 0
        out = []
        chunk_size = 10000

        kH, kW = self.weight.shape[-2:]
        C_out, out_H, out_W = self.out_shape
        C_in, in_H, in_W = self.in_shape

        base_H = (out_H - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (kH - 1) + 1
        base_W = (out_W - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (kW - 1) + 1
        op_h = in_H - base_H
        op_w = in_W - base_W

        if not (0 <= op_h < self.stride[0] and 0 <= op_w < self.stride[1]):
            raise ValueError(
                f"Computed output_padding {(op_h, op_w)} invalid for stride {self.stride}. "
                f"Check in/out shapes or conv params."
            )

        out_padding = (int(op_h), int(op_w))

        while i < v.size(0):
            out.append(
                F.conv_transpose2d(
                    v[i:min(i + chunk_size, v.size(0))],
                    self.weight,
                    None,
                    self.stride,
                    self.padding,
                    out_padding,
                    self.group,
                    self.dilation,
                )
            )
            i += chunk_size

        return torch.cat(out, 0)

    def conv_reshape(self, v):
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
            raise ValueError(f"Expected v to have 1 or 2 dims, got {v.dim()}")

    def T(self, inp_As):
        inp_A = inp_As[-1]
        if inp_A is None:
            return None

        inp_A_4d, batched = self.conv_reshape(inp_A)
        new_inp_A_4d = self.conv_transpose2d(inp_A_4d)
        flat = new_inp_A_4d.view(new_inp_A_4d.size(0), -1)

        if batched:
            return flat
        else:
            return flat.view(-1)

    def objective(self, A):
        if self.bias is None:
            return torch.tensor(0.0, device=A.device, dtype=A.dtype)

        bias_flat = self.bias.view(-1)

        if A.dim() == 1:
            return -A.matmul(bias_flat)
        elif A.dim() == 2:
            return -(A * bias_flat.unsqueeze(0)).sum(dim=1)
        else:
            raise ValueError(f"Expected A to have 1 or 2 dims, got {A.dim()}")


class DualFlatten_Ind:
    def __init__(self):
        pass

    def T(self, inp_As):
        # Flatten is identity in backward coefficient space if everything is already vectorized.
        return inp_As[-1]

    def objective(self, A):
        return torch.tensor(0.0, device=A.device, dtype=A.dtype)


class DualRelu_Ind:
    """
    ReLU dual layer with optional alpha lower slope.

    If alpha is provided:
        unstable lower slope = clamp(alpha, 0, 1)

    If alpha is None:
        - relu_precise=False: lower slope on unstable uses upper secant slope
        - relu_precise=True : lower slope uses 0/1 heuristic
    """

    def __init__(self, inp_lb, inp_ub, relu_precise=False, alpha=None):
        self.inp_lb = inp_lb
        self.inp_ub = inp_ub
        self.positive_mask = (inp_lb >= 0) & (inp_ub > 0)
        self.negative_mask = (inp_ub <= 0)
        self.unstable_mask = (~self.positive_mask) & (~self.negative_mask)
        self.relu_precise = relu_precise
        self.alpha = alpha

    def get_lambda(self, inp_A):
        inp_lb = self.inp_lb
        inp_ub = self.inp_ub

        base_lam = torch.zeros_like(inp_lb)
        base_lam = torch.where(self.positive_mask, torch.ones_like(inp_lb), base_lam)

        unstable = self.unstable_mask
        upper_slope = inp_ub / (inp_ub - inp_lb + 1e-15)

        if self.alpha is not None:
            lower_slope = torch.clamp(self.alpha, 0.0, 1.0)
        else:
            if self.relu_precise:
                lower_slope = torch.where(
                    inp_ub > -inp_lb,
                    torch.ones_like(inp_lb),
                    torch.zeros_like(inp_lb),
                )
            else:
                # relaxed mode: use secant
                lower_slope = upper_slope

        if inp_A.dim() == 1:
            lam = base_lam.clone()
            lam = torch.where(unstable & (inp_A >= 0), upper_slope, lam)
            lam = torch.where(unstable & (inp_A < 0), lower_slope, lam)
            return lam

        elif inp_A.dim() == 2:
            batch_size = inp_A.size(0)
            lam = base_lam.unsqueeze(0).expand(batch_size, -1).clone()

            unstable_b = unstable.unsqueeze(0)
            upper_slope_b = upper_slope.unsqueeze(0)
            lower_slope_b = lower_slope.unsqueeze(0)

            lam = torch.where(unstable_b & (inp_A >= 0), upper_slope_b, lam)
            lam = torch.where(unstable_b & (inp_A < 0), lower_slope_b, lam)
            return lam

        else:
            raise ValueError(f"Expected inp_A to have 1 or 2 dims, got {inp_A.dim()}")

    def T(self, inp_As):
        inp_A = inp_As[-1]
        if inp_A is None:
            return None

        lam = self.get_lambda(inp_A)
        new_inp_A = lam * inp_A
        return new_inp_A

    def objective(self, A):
        """
        Intercept contribution from ReLU upper relaxation.

        For unstable ReLU:
            y <= lambda_u x + mu_u
            mu_u = -u l / (u-l)

        In the dual objective, positive coefficients contribute this intercept.
        """
        temp_coef = (self.inp_ub * self.inp_lb) / (self.inp_ub - self.inp_lb + 1e-15)
        coef = torch.where(self.unstable_mask, temp_coef, torch.zeros_like(temp_coef))

        if A.dim() == 1:
            pos_A = torch.clamp(A, min=0)
            return pos_A.matmul(coef)

        elif A.dim() == 2:
            pos_A = torch.clamp(A, min=0)
            return (pos_A * coef.unsqueeze(0)).sum(dim=1)

        else:
            raise ValueError(f"Expected A to have 1 or 2 dims, got {A.dim()}")