from experiments import mnist

""" helper """
def subset_by_original_indices(points, selected_indices):
    selected_set = set(selected_indices)

    keep_pos = [
        i for i, orig_idx in enumerate(points.indices)
        if orig_idx in selected_set
    ]

    subset = points.copy()
    subset.images = points.images[keep_pos]
    subset.labels = points.labels[keep_pos]
    subset.indices = [points.indices[i] for i in keep_pos]

    return subset


""" collect input points """
def clean_points(repair_label, dnn, device, dtype):
    testset = mnist.datasets.Dataset('identity', 'test').reshape(784).to(device=device, dtype=dtype)
    pos_clean_points, neg_clean_points = testset.filter_label(repair_label).misclassified(dnn)
    return pos_clean_points, neg_clean_points

def damaged_points(repair_label, dnn, device, dtype):
    mnist_c = mnist.datasets.MNIST_C(corruption='fog', split='test').reshape(784).to(device=device, dtype=dtype)
    pos_damaged_points, neg_damaged_points = mnist_c.filter_label(repair_label).misclassified(dnn)
    return pos_damaged_points, neg_damaged_points

def neg_damaged_points_from_pos_clean(pos_clean_points, neg_damaged_points):
    pos_clean_and_neg_damaged_indices = sorted(set(pos_clean_points.indices) & set(neg_damaged_points.indices))
    repaired_points = subset_by_original_indices(neg_damaged_points, pos_clean_and_neg_damaged_indices)
    clean_original_points = subset_by_original_indices(pos_clean_points, pos_clean_and_neg_damaged_indices)
    return pos_clean_and_neg_damaged_indices, repaired_points, clean_original_points

def repair_points(repair_label, dnn, device, dtype):
    pos_clean_points, _ = clean_points(repair_label, dnn, device, dtype)
    _, neg_damaged_points = damaged_points(repair_label, dnn, device, dtype)
    print(sorted(set(pos_clean_points.indices) & set(neg_damaged_points.indices)))
    repaired_indices, repaired_points, base_points = neg_damaged_points_from_pos_clean(pos_clean_points, neg_damaged_points)
    return repaired_indices, repaired_points, base_points