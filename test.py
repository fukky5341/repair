from input_space.divide_sets import make_root_region, explore_input_space
from input_space.generate_input import damaged_points
from experiments import mnist
import sytorch as st

device = 'cpu'
dtype = st.float64

dnn = mnist.model('9x100').to(dtype=dtype).to(device=device)
target_label = 8
pos_points, neg_points = damaged_points(target_label, dnn)

x_buggy = neg_points.images[0].unsqueeze(0)  # shape: (1, 784)
y_buggy = neg_points.labels[0].item()        # scalar

root_region = make_root_region(x_buggy=x_buggy, eps=0.05, target_label=y_buggy)

# positive_regions, negative_regions, undecided_regions = explore_input_space(
#     model=dnn,
#     root_region=root_region,
#     num_classes=10,
#     max_depth=8,
#     max_regions=200,
#     bound_method="CROWN-IBP",
# )