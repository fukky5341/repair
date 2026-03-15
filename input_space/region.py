from dataclasses import dataclass, field
from enum import Enum
import torch
from typing import Optional
from repair.util import Spec


class RegionStatus(Enum):
    unprocessed = "unprocessed"
    positive = "positive"
    negative = "negative"
    undecided = "undecided"


@dataclass
class Region:
    center_point: torch.Tensor  # shape: (1, C, H, W) or (1, D)
    lb: torch.Tensor          # shape: (1, C, H, W) or (1, D)
    ub: torch.Tensor          # same shape as lb
    target_label: int
    data_id: int
    depth: int = 0

    # specification
    spec: Optional[Spec] = None

    # filled after analysis
    status: RegionStatus = RegionStatus.unprocessed
    margin_lbs: Optional[torch.Tensor] = None   # shape: (num_classes-1,)
    candidate_x: Optional[torch.Tensor] = None
    violated_label: Optional[int] = None
    score: Optional[float] = None


    def add_spec(self, num_classes, target_label, device, dtype):
        '''
        c_i = y_t - y_i for i != t
        C = [c_1, c_2, ..., c_{t-1}, c_{t+1}, ..., c_n]  # shape: (num_constraints, num_outputs)
        '''
        C = torch.eye(num_classes, device=device, dtype=dtype)
        C = C[target_label:target_label+1] - torch.cat([C[:target_label], C[target_label+1:]], dim=0)
        self.spec = Spec(C=C, target_label=target_label)