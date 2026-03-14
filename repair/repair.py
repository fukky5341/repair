import torch
import torch.nn as nn

from experiments import mnist
import sytorch as st

from input_space.generate_input import ( repair_regions )
from .subspace import ( build_safe_subspaces )


def repair_params_by_lp():
    # lp_solver = ...
    # ...
    # new_params = ...
    # returns new_params
    pass


def update_repaired_layer():
    # replace repaired layer with new parameters
    pass


def evaluate_repaired_dnn():
    # evaluate the repaired DNN on repaired regions
    pass


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
    # subspace
    safe_subspaces = build_safe_subspaces(dnn, repaired_layer_idx, base_regions)
    # get repaired parameters by solving the lp model
    #todo: repaired_params = repair_params_by_lp(solver)
    # update the repaired layer with repaired parameters
    #todo: update_repaired_layer(dnn, repaired_layer_idx, repaired_params)
    # evaluate the repaired DNN on repaired regions
    is_repaired = False
    #todo: evaluate_repaired_dnn(dnn, repaired_regions)
    if is_repaired:
        break
    else:
        continue
