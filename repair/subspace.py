from dataclasses import dataclass
import torch

from repair.bound import ( get_concrete_bounds, check_violation )


'''
Build subspace that under-approximates the feasible region.
If the output of the repaired layer is in the subspace, 
then the repaired DNN is guaranteed to satisfy the specification.

network: N
subspace for k-th layer: S^(k)
S^(k) is a subspace of N[k:], i.e., N[:k](x) \in S^(k) => N(x) \in OutSpec(x)

Assumption:
    given:
        - positive regions Ps (e.g., v-polytopes)
        - network N with alternating linear and ReLU layers
        - repaired layer index k and its type is linear

From here on, we take one P to make description simpler.

Steps:
- compute concrete bounds N[:k+1](P) --> lb, ub
- initialize the subspace S^(k) with the bounds lb and ub
- get gradient to enlarge the subspace S^(k)
    - gradient can be the direction towards the violated region
- itereatively enlarge the subspace S^(k) until it meet violating points (e.g., lb' -= subsp_lr*gradient, ub' += subsp_lr*gradient)
    - compute the output of N[k:](x) for x in [lb', ub'] to check whether it meets violating points
'''

@dataclass
class Subspace:
    repaired_layer_idx: int
    lb: torch.Tensor
    ub: torch.Tensor
    data_id: int


# ==== Debug ====
def debug_subspace(stage, lb, ub, netS, spec, step=None):
    """
    Print useful information about the current subspace.
    """

    center = (lb + ub) / 2
    width = ub - lb

    is_violated = check_violation(netS, lb, ub, spec)

    print("\n[Subspace Debug]")
    print(f"stage: {stage}")
    if step is not None:
        print(f"iter: {step}")

    print(f"center norm: {center.norm().item():.6f}")
    print(f"avg width: {width.mean().item():.6f}")
    print(f"max width: {width.max().item():.6f}")
    print(f"min width: {width.min().item():.6f}")

    print(f"violation: {is_violated}")

    # show first few dimensions
    k = min(5, lb.shape[0])
    print("lb[:5]:", lb[:k].detach().cpu().numpy())
    print("ub[:5]:", ub[:k].detach().cpu().numpy())


# ==== Update subspace ====
# --- helper ---
def scale_box(lb, ub, alpha):
    center = (lb + ub) / 2
    width = (ub - lb) / 2
    new_lb = center - alpha * width
    new_ub = center + alpha * width
    return new_lb, new_ub


def maximal_uniform_scale(netS, lb, ub, spec, alpha_max=10.0, iters=20):

    low = 1.0
    high = alpha_max
    best = 1.0

    for _ in range(iters):

        mid = (low + high) / 2

        new_lb, new_ub = scale_box(lb, ub, mid)

        if check_violation(netS, new_lb, new_ub, spec):
            high = mid
        else:
            best = mid
            low = mid

    return best

def per_dimension_scale(netS, lb, ub, spec, alpha_max=10.0, iters=15):
    ''' Binary search for each dimension to find the maximal scaling factor '''

    center = (lb + ub) / 2
    width = (ub - lb) / 2

    dim = lb.shape[0]
    alpha = torch.ones_like(lb)

    for i in range(dim):

        low = 1.0
        high = alpha_max
        best = 1.0

        for _ in range(iters):
            mid = (low + high) / 2

            test_lb = center.clone()
            test_ub = center.clone()

            test_lb[i] = center[i] - mid * width[i]
            test_ub[i] = center[i] + mid * width[i]

            if check_violation(netS, test_lb, test_ub, spec):
                high = mid
            else:
                best = mid
                low = mid

        alpha[i] = best

    new_lb = center - alpha * width
    new_ub = center + alpha * width

    return new_lb, new_ub

def violation_and_grad(netS, x, spec):

    x = x.clone().detach().requires_grad_(True)
    y = netS(x.unsqueeze(0)).squeeze(0)

    loss = spec.violation_loss(y)
    loss.backward()
    grad = x.grad.detach()

    return loss.detach(), grad

def gradient_expand(netS, lb, ub, spec,
                    subsp_lr=1e-2,
                    max_iters=20,
                    subsp_lr_decay=0.5):

    subsp_lb = lb.clone()
    subsp_ub = ub.clone()

    for _ in range(max_iters):

        center = (subsp_lb + subsp_ub) / 2

        loss, grad = violation_and_grad(netS, center, spec)

        if loss.item() >= 0:
            break

        direction = grad.abs()
        direction = direction / (direction.max() + 1e-8)

        expand = subsp_lr * direction
        new_lb = subsp_lb - expand
        new_ub = subsp_ub + expand

        if check_violation(netS, new_lb, new_ub, spec):
            subsp_lr *= subsp_lr_decay
            if subsp_lr < 1e-5:
                break
            continue

        subsp_lb = new_lb
        subsp_ub = new_ub

    return subsp_lb, subsp_ub

def update_subspace(netF, netS, P, 
                    subsp_lr=1e-2, max_iters=20, subsp_lr_decay=0.5,
                    use_gradient=True, use_uniform_scale=True, use_per_dim_scale=False,
                    # use_gradient=False, use_uniform_scale=False, use_per_dim_scale=True,
                    debug=True):
    # input bounds for netF
    inF_lb, inF_ub = P.lb.clone(), P.ub.clone()
    # compute concrete bounds for the positive region
    lb, ub = get_concrete_bounds(netF, inF_lb, inF_ub)
    # check violation for initial subspace
    inS_lb, inS_ub = lb[-1].clone(), ub[-1].clone()  # input bounds for netS

    if debug:
        debug_subspace("initial", inS_lb, inS_ub, netS, P.spec)

    if check_violation(netS, inS_lb, inS_ub, P.spec):
        # if violated
        return inS_lb, inS_ub  # return the initial subspace (no enlargement)
    
    # initialize the subspace with the bounds
    sub_lb, sub_ub = inS_lb.clone(), inS_ub.clone()

    # ---- Stage 1: gradient expansion ----
    if use_gradient:
        sub_lb, sub_ub = gradient_expand(netS, sub_lb, sub_ub, P.spec)

        if debug:
            debug_subspace("after gradient expansion", sub_lb, sub_ub, netS, P.spec)

    # ---- Stage 2: uniform scaling ----
    if use_uniform_scale:
        keep_lb, keep_ub = sub_lb.clone(), sub_ub.clone()
        alpha = maximal_uniform_scale(netS, sub_lb, sub_ub, P.spec)
        sub_lb, sub_ub = scale_box(sub_lb, sub_ub, alpha)
        is_violated = check_violation(netS, sub_lb, sub_ub, P.spec)
        if is_violated:
            sub_lb, sub_ub = keep_lb, keep_ub  # revert to the last non-violating subspace

        if debug:
            debug_subspace("after uniform scaling", sub_lb, sub_ub, netS, P.spec, step=alpha)

    # ---- Stage 3: per-dimension scaling ----
    if use_per_dim_scale:
        sub_lb, sub_ub = per_dimension_scale(netS, sub_lb, sub_ub, P.spec)

        if debug:
            debug_subspace("after per-dimension scaling", sub_lb, sub_ub, netS, P.spec)

    return sub_lb, sub_ub



# ==== Build subspace ====
def build_safe_subspaces(net, repaired_layer: int, positive_regions):
    '''
    repaired_layer (k): index of the repaired affine layer (Li)
    net: [L0, R1, L2, R3, ..., Rk-1, Lk, Rk, Lk+1, Rk+2, ...]
    netF = net[:k+1]: [L0, R1, L2, R3, ..., Rk-1, Lk]
    netS = net[k+1:]: [Rk+2, ...]
    '''
    netF = net[:repaired_layer+1]
    netS = net[repaired_layer+1:]
    # For each regions, compute subspace
    safe_subspaces = []
    for P in positive_regions:
        subsp_lb, subsp_ub = update_subspace(netF, netS, P)
        subspace = Subspace(repaired_layer, subsp_lb, subsp_ub, P.data_id)
        safe_subspaces.append(subspace)
    
    return safe_subspaces
