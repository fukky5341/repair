from experiments import mnist
import torch
import numpy as np
import copy
from .region import Region


def _normalize_positions(keep_pos):
    if isinstance(keep_pos, torch.Tensor):
        keep_pos = keep_pos.detach().cpu().tolist()
    elif isinstance(keep_pos, np.ndarray):
        keep_pos = keep_pos.tolist()
    return [int(i) for i in keep_pos]


def _subset_field(field, keep_pos):
    if isinstance(field, torch.Tensor):
        idx = torch.tensor(keep_pos, device=field.device, dtype=torch.long)
        return field.index_select(0, idx).clone()
    elif isinstance(field, np.ndarray):
        return field[np.array(keep_pos, dtype=np.int64)].copy()
    else:
        return [field[i] for i in keep_pos]


def subset_by_positions(points, keep_pos):
    """
    Keep entries by local positions inside the current dataset object.
    """
    keep_pos = _normalize_positions(keep_pos)

    # IMPORTANT: use deepcopy, not points.copy()
    subset = copy.deepcopy(points)

    subset.images = _subset_field(points.images, keep_pos)
    subset.labels = _subset_field(points.labels, keep_pos)

    if hasattr(points, "indices") and points.indices is not None:
        subset.indices = _subset_field(points.indices, keep_pos)

    return subset


def subset_by_original_indices(points, selected_indices):
    selected_set = set(int(idx) for idx in selected_indices)

    keep_pos = [
        i for i, orig_idx in enumerate(points.indices)
        if int(orig_idx) in selected_set
    ]

    return subset_by_positions(points, keep_pos)


def filter_by_label(points, target_label):
    if target_label is None:
        return points

    if isinstance(points.labels, torch.Tensor):
        keep_pos = (points.labels == target_label).nonzero(as_tuple=True)[0]
    elif isinstance(points.labels, np.ndarray):
        keep_pos = np.where(points.labels == target_label)[0]
    else:
        keep_pos = [i for i, y in enumerate(points.labels) if int(y) == target_label]

    return subset_by_positions(points, keep_pos)


@torch.no_grad()
def split_by_misclassification(points, dnn):
    logits = dnn(points.images)
    preds = logits.argmax(dim=1)

    if isinstance(points.labels, torch.Tensor):
        labels = points.labels.to(device=preds.device, dtype=preds.dtype)
    elif isinstance(points.labels, np.ndarray):
        labels = torch.from_numpy(points.labels).to(device=preds.device, dtype=preds.dtype)
    else:
        labels = torch.tensor(points.labels, device=preds.device, dtype=preds.dtype)

    labels = labels.reshape(-1).to(dtype=preds.dtype)

    correct_mask = (preds == labels)
    wrong_mask = ~correct_mask

    pos_idx = correct_mask.nonzero(as_tuple=True)[0]
    neg_idx = wrong_mask.nonzero(as_tuple=True)[0]

    pos_points = subset_by_positions(points, pos_idx)
    neg_points = subset_by_positions(points, neg_idx)

    return pos_points, neg_points


def clean_points(device, dtype):
    return (
        mnist.datasets.Dataset('identity', 'test')
        .reshape(784)
        .to(device=device, dtype=dtype)
    )


def damaged_points(device, dtype):
    return (
        mnist.datasets.MNIST_C(corruption='fog', split='test')
        .reshape(784)
        .to(device=device, dtype=dtype)
    )


def neg_damaged_points_from_pos_clean(pos_clean_points, neg_damaged_points):
    """
    Match by original dataset index, and force the same ordering in both outputs.
    """
    clean_map = {int(idx): i for i, idx in enumerate(pos_clean_points.indices)}
    damaged_map = {int(idx): i for i, idx in enumerate(neg_damaged_points.indices)}

    common_indices = sorted(set(clean_map.keys()) & set(damaged_map.keys()))

    clean_pos = [clean_map[idx] for idx in common_indices]
    damaged_pos = [damaged_map[idx] for idx in common_indices]

    clean_original_points = subset_by_positions(pos_clean_points, clean_pos)
    repaired_points = subset_by_positions(neg_damaged_points, damaged_pos)

    # overwrite indices with aligned original indices
    clean_original_points.indices = common_indices
    repaired_points.indices = common_indices

    return common_indices, repaired_points, clean_original_points


def repair_points(repair_label, dnn, device, dtype):
    c_points = clean_points(device, dtype)
    d_points = damaged_points(device, dtype)

    if repair_label is not None:
        f_c_points = filter_by_label(c_points, repair_label)
        f_d_points = filter_by_label(d_points, repair_label)
    else:
        f_c_points = c_points
        f_d_points = d_points

    pos_clean_points, neg_clean_points = split_by_misclassification(f_c_points, dnn)
    pos_damaged_points, neg_damaged_points = split_by_misclassification(f_d_points, dnn)

    repaired_indices, repaired_points, base_points = \
        neg_damaged_points_from_pos_clean(pos_clean_points, neg_damaged_points)

    return repaired_indices, repaired_points, base_points

def repair_regions(repair_label, dnn, num_regions, neg_eps, pos_eps, device, dtype):
    repaired_indices, repaired_points, base_points = repair_points(repair_label, dnn, device, dtype)

    repaired_regions = []
    base_regions = []
    for i in range(num_regions):
        repaired_region = Region(
            center_point = repaired_points.images[i].unsqueeze(0),
            lb = repaired_points.images[i].unsqueeze(0) - neg_eps,
            ub = repaired_points.images[i].unsqueeze(0) + neg_eps,
            target_label = repaired_points.labels[i].item(),
            data_id=repaired_indices[i]
        )
        base_region = Region(
            center_point = base_points.images[i].unsqueeze(0),
            lb = base_points.images[i].unsqueeze(0) - pos_eps,
            ub = base_points.images[i].unsqueeze(0) + pos_eps,
            target_label = base_points.labels[i].item(),
            data_id=repaired_indices[i]
        )
        # add spec
        num_classes = dnn(base_points.images[:1]).shape[1]  # assuming output shape is (1, num_classes)
        repaired_region.add_spec(num_classes, repaired_region.target_label)
        base_region.add_spec(num_classes, base_region.target_label)

        repaired_regions.append(repaired_region)
        base_regions.append(base_region)

    return repaired_regions, base_regions