import numpy as np
import torch
from dataclasses import dataclass
from collections import deque
from typing import Optional
# APRNN imports
from experiments.mnist.datasets import Dataset
# auto_LiRPA imports
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


# =========================================================
# Region
# =========================================================

@dataclass
class Region:
    lb: torch.Tensor
    ub: torch.Tensor
    target_label: int
    depth: int = 0
    region_id: Optional[int] = None
    parent_id: Optional[int] = None

    status: str = "unprocessed"   # "positive", "negative", "undecided"
    margin_lbs: Optional[torch.Tensor] = None
    candidate_x: Optional[torch.Tensor] = None
    violated_label: Optional[int] = None
    score: Optional[float] = None


def make_root_region(x_buggy: torch.Tensor, eps: float, target_label: int) -> Region:
    lb = torch.clamp(x_buggy - eps, 0.0, 1.0)
    ub = torch.clamp(x_buggy + eps, 0.0, 1.0)
    return Region(lb=lb, ub=ub, target_label=target_label, depth=0)

def make_root_region_set(inp_points: Dataset, eps: float, device):
    root_regions = []
    for i in range(len(inp_points)):
        x_buggy = inp_points.images[i].unsqueeze(0).to(device)  # shape: (1, 784)
        y_buggy = inp_points.labels[i].item()        # scalar
        root_region = make_root_region(x_buggy=x_buggy, eps=eps, target_label=y_buggy)
        root_regions.append(root_region)
    return root_regions


# =========================================================
# auto_LiRPA helpers
# =========================================================

def make_bounded_model(model: torch.nn.Module, input_shape: torch.Size, device):
    param = next(model.parameters())
    dummy = torch.empty(input_shape, device=device, dtype=param.dtype)
    return BoundedModule(model, dummy, device=device)


def build_all_margin_specs(
    target_label: int,
    num_classes: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[int]]:
    """
    Build C so that (C @ logits) gives margins f_t - f_i for all i != t.

    Returns:
        C: shape (batch_size, num_classes - 1, num_classes)
        other_labels: list of labels i != target_label in row order
    """
    other_labels = [i for i in range(num_classes) if i != target_label]
    num_specs = len(other_labels)

    C = torch.zeros(
        (batch_size, num_specs, num_classes),
        device=device,
        dtype=dtype,
    )
    C[:, :, target_label] = 1.0
    for row_idx, i in enumerate(other_labels):
        C[:, row_idx, i] = -1.0

    return C, other_labels


def compute_margin_lower_bounds_reuse(
    bounded_model: BoundedModule,
    lb: torch.Tensor,
    ub: torch.Tensor,
    target_label: int,
    num_classes: int,
    method: str = "CROWN",
) -> tuple[torch.Tensor, list[int]]:
    assert lb.shape == ub.shape
    assert lb.ndim >= 2
    assert torch.all(lb <= ub), "Found entries with lb > ub."

    # Make everything match the model dtype
    param_dtype = next(bounded_model.parameters()).dtype
    lb = lb.to(dtype=param_dtype)
    ub = ub.to(dtype=param_dtype)
    device = lb.device

    x_center = 0.5 * (lb + ub)
    ptb = PerturbationLpNorm(norm=np.inf, x_L=lb, x_U=ub)
    x_bounded = BoundedTensor(x_center, ptb)

    C, other_labels = build_all_margin_specs(
        target_label=target_label,
        num_classes=num_classes,
        batch_size=1,
        device=device,
        dtype=lb.dtype,
    )

    if method == "alpha-CROWN":
        bounded_model.set_bound_opts({
            "optimize_bound_args": {
                "iteration": 20,
                "lr_alpha": 0.1,
            }
        })

    # # debugging: print dtypes to check for mismatches
    # print("lb dtype:", lb.dtype)
    # print("ub dtype:", ub.dtype)
    # print("x_center dtype:", x_center.dtype)
    # print("C dtype:", C.dtype)
    # print("model dtype:", next(bounded_model.parameters()).dtype)

    margin_lbs, margin_ubs = bounded_model.compute_bounds(
        x=(x_bounded,),
        C=C,
        method=method,
    )
    return margin_lbs, other_labels


# =========================================================
# Search / verification helpers
# =========================================================

@torch.no_grad()
def check_real_violation(model: torch.nn.Module, x: torch.Tensor, target_label: int) -> bool:
    logits = model(x)
    pred = int(torch.argmax(logits, dim=1).item())
    return pred != target_label


def search_counterexample(
    model: torch.nn.Module,
    lb: torch.Tensor,
    ub: torch.Tensor,
    target_label: int,
    margin_lbs: torch.Tensor,   # shape: (num_classes - 1,)
    num_classes: int,
    steps: int = 50,
    step_size: float = 1e-2,
    restarts: int = 5,
) -> tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Heuristic PGD-style search for a real violating point.

    Returns:
        candidate_x, violated_label
    """
    device = lb.device
    dtype = lb.dtype

    _, labels_wo_target = build_all_margin_specs(
        target_label=target_label,
        num_classes=num_classes,
        batch_size=1,
        device=device,
        dtype=dtype,
    )

    worst_idx = int(torch.argmin(margin_lbs).item())
    violated_label = labels_wo_target[worst_idx]

    best_x = None
    best_margin = float("inf")

    for _ in range(restarts):
        x = lb + torch.rand_like(lb) * (ub - lb)
        x = x.detach().clone().requires_grad_(True)

        for _ in range(steps):
            logits = model(x)
            margin = logits[0, target_label] - logits[0, violated_label]

            model.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad.zero_()
            margin.backward()

            with torch.no_grad():
                x = x - step_size * x.grad.sign()
                x = torch.max(torch.min(x, ub), lb)

            x.requires_grad_(True)

        with torch.no_grad():
            logits = model(x)
            final_margin = (logits[0, target_label] - logits[0, violated_label]).item()

        if final_margin < best_margin:
            best_margin = final_margin
            best_x = x.detach().clone()

    return best_x, violated_label


def split_region(region: Region) -> tuple[Optional[Region], Optional[Region]]:
    """
    Binary split along the widest dimension.
    Returns (None, None) if no nontrivial split is possible.
    """
    lb = region.lb.clone()
    ub = region.ub.clone()

    flat_lb = lb.view(-1)
    flat_ub = ub.view(-1)
    widths = flat_ub - flat_lb

    max_width = widths.max().item()
    if max_width <= 1e-12:
        return None, None

    split_dim = int(torch.argmax(widths).item())
    mid = 0.5 * (flat_lb[split_dim] + flat_ub[split_dim])

    lb1, ub1 = flat_lb.clone(), flat_ub.clone()
    lb2, ub2 = flat_lb.clone(), flat_ub.clone()

    ub1[split_dim] = mid
    lb2[split_dim] = mid

    child1 = Region(
        lb=lb1.view_as(lb),
        ub=ub1.view_as(ub),
        target_label=region.target_label,
        depth=region.depth + 1,
        parent_id=region.region_id,
    )
    child2 = Region(
        lb=lb2.view_as(lb),
        ub=ub2.view_as(ub),
        target_label=region.target_label,
        depth=region.depth + 1,
        parent_id=region.region_id,
    )
    return child1, child2


def analyze_region(
    model: torch.nn.Module,
    bounded_model: BoundedModule,
    region: Region,
    num_classes: int,
    bound_method: str = "CROWN",
) -> Region:
    """
    Analyze one region and classify it as:
      - positive: all points are certified correct
      - negative: a real counterexample was found
      - undecided: neither of the above
    """
    margin_lbs, other_labels = compute_margin_lower_bounds_reuse(
        bounded_model=bounded_model,
        lb=region.lb,
        ub=region.ub,
        target_label=region.target_label,
        num_classes=num_classes,
        method=bound_method,
    )

    # Assume batch size = 1 for region exploration.
    margin_lbs_1d = margin_lbs[0]
    region.margin_lbs = margin_lbs_1d.detach().clone()

    # Positive: all target margins are nonnegative.
    if torch.all(margin_lbs_1d >= 0):
        region.status = "positive"
        region.score = float(margin_lbs_1d.min().item())
        return region

    # Try to find a concrete violating point.
    candidate_x, violated_label = search_counterexample(
        model=model,
        lb=region.lb,
        ub=region.ub,
        target_label=region.target_label,
        margin_lbs=margin_lbs_1d,
        num_classes=num_classes,
    )

    if candidate_x is not None:
        is_real_bug = check_real_violation(
            model=model,
            x=candidate_x,
            target_label=region.target_label,
        )
        if is_real_bug:
            region.status = "negative"
            region.candidate_x = candidate_x.detach().clone()
            region.violated_label = violated_label
            region.score = float((-margin_lbs_1d.min()).item())
            return region

    region.status = "undecided"
    region.candidate_x = candidate_x
    region.violated_label = violated_label
    region.score = float((-margin_lbs_1d.min()).item())
    return region


# =========================================================
# Main exploration loop
# =========================================================

def explore_input_space(
    model: torch.nn.Module,
    root_region: Region,
    num_classes: int,
    max_depth: int = 10,
    max_regions: int = 1000,
    bound_method: str = "CROWN",  # "IBP" or "CROWN-IBP" or "CROWN" or "alpha-CROWN"
):
    """
    Explore the input region and partition the current leaves into:
        positive_regions, negative_regions, undecided_regions
    """
    bounded_model = make_bounded_model(
        model=model,
        input_shape=root_region.lb.shape,
        device=root_region.lb.device,
    )

    queue = deque([root_region])
    positive_regions = []
    negative_regions = []
    undecided_regions = []

    next_region_id = 0
    processed = 0

    while queue and processed < max_regions:
        region = queue.popleft()
        if region.region_id is None:
            region.region_id = next_region_id
            next_region_id += 1

        analyzed = analyze_region(
            model=model,
            bounded_model=bounded_model,
            region=region,
            num_classes=num_classes,
            bound_method=bound_method,
        )
        processed += 1

        if analyzed.status == "positive":
            positive_regions.append(analyzed)
            continue

        if analyzed.status == "negative":
            negative_regions.append(analyzed)
            continue

        # undecided
        if analyzed.depth >= max_depth:
            undecided_regions.append(analyzed)
            continue

        child1, child2 = split_region(analyzed)
        if child1 is None or child2 is None:
            undecided_regions.append(analyzed)
            continue

        queue.append(child1)
        queue.append(child2)

    # Remaining queued regions become undecided if budget is exhausted.
    while queue:
        undecided_regions.append(queue.popleft())

    return positive_regions, negative_regions, undecided_regions