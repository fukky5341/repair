import torch
import torch.nn as nn

from experiments import mnist
import sytorch as st

from input_space.generate_input import repair_regions
from LPsolver.solver import LPSolver
from .subspace import build_safe_subspaces
from .bound import ( get_concrete_bounds, check_violation )


def repair_params_by_lp(dnn, repaired_layer_idx, repaired_regions, base_regions, safe_subspaces):
    layer = dnn[repaired_layer_idx]
    solver = LPSolver(weight=layer.weight, bias=layer.bias)
    solver.add_sign_constraints()
    # add constraints for base regions
    for n_region, p_region, subspace in zip(repaired_regions, base_regions, safe_subspaces):
        assert n_region.data_id == p_region.data_id == subspace.data_id
        # compute the input bounds for the repaired layer
        fixed_net = dnn[:repaired_layer_idx]
        zlb, zub = get_concrete_bounds(fixed_net, p_region.lb, p_region.ub)
        # add constraints to ensure the output of the base region is in the safe subspace
        solver.add_region(zlb, zub, subspace.lb, subspace.ub)
        # set the objective to minimize the distance between the repaired region and the safe subspace
        solver.set_objective(n_region, subspace)
    
    solver.build_objective()
    repaired_params = solver.solve()

    return repaired_params


def update_repaired_layer(dnn, layer_idx, new_params):
    new_W, new_B = new_params
    layer = dnn[layer_idx]

    layer.weight.data = new_W.to(layer.weight.device)
    layer.bias.data = new_B.to(layer.bias.device)


def evaluate_repaired_dnn(dnn, repaired_layer_idx, repaired_regions, base_regions, subspaces):
    # evaluate the repaired DNN
    # Note: this evaluation is still approximate
    for n_region, p_region, subspace in zip(repaired_regions, base_regions, subspaces):
        assert n_region.data_id == p_region.data_id == subspace.data_id
        n_lb, n_ub = n_region.lb, n_region.ub
        p_lb, p_ub = p_region.lb, p_region.ub

        # check by subspace
        sub_net = dnn[:repaired_layer_idx+1]
        n_is_violated = check_violation(sub_net, n_lb, n_ub, n_region.spec)
        if n_is_violated:
            return False
        p_is_violated = check_violation(sub_net, p_lb, p_ub, p_region.spec)
        if p_is_violated:
            raise ValueError("can happen due to the approximation? I think it should not happen.")

        # check by using whole repaired DNN
        n_is_violated = check_violation(dnn, n_lb, n_ub, n_region.spec)
        if n_is_violated:
            return False
        p_is_violated = check_violation(dnn, p_lb, p_ub, p_region.spec)
        if p_is_violated:
            raise ValueError("can happen due to the approximation? I think it should not happen.")




# Set parameters
device = 'cpu'
dtype = st.float64

# Load network
dnn = mnist.model('9x100').to(dtype=dtype).to(device=device)

# Repair parameters
# Target label
target_label = 8
# The number of v-polytopes
num_v_polys = 3
# perturbation distance
inp_eps = 0.01
# repaired layer index
repaired_layer_idx_set = (i for i in range(1,16) if i % 2 == 0)  # even indices correspond to Linear layers

# Repaired regions and corresponding Base (positive) regions
repaired_regions, base_regions = repair_regions(
    repair_label=target_label,
    dnn=dnn,
    num_regions=num_v_polys,
    neg_eps=inp_eps,
    pos_eps=inp_eps,
    device=device,
    dtype=dtype
)

# --- Repair ---
for repaired_layer_idx in repaired_layer_idx_set:
    # --- collect safe subspaces ---
    safe_subspaces = build_safe_subspaces(dnn, repaired_layer_idx, base_regions)
    # --- get repaired parameters by solving the lp model ---
    repaired_params = repair_params_by_lp(dnn, repaired_layer_idx, 
                                          repaired_regions=repaired_regions, 
                                          base_regions=base_regions, 
                                          safe_subspaces=safe_subspaces)
    # update the repaired layer with repaired parameters
    update_repaired_layer(dnn, repaired_layer_idx, repaired_params)
    # evaluate the repaired DNN on repaired regions
    is_repaired = evaluate_repaired_dnn(dnn, repaired_layer_idx, repaired_regions, base_regions, safe_subspaces)
    if is_repaired:
        break
    else:
        continue
