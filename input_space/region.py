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
    depth: int = 0
    data_id: Optional[int] = None

    # specification
    spec: Optional[Spec] = None

    # filled after analysis
    status: RegionStatus = RegionStatus.unprocessed
    margin_lbs: Optional[torch.Tensor] = None   # shape: (num_classes-1,)
    candidate_x: Optional[torch.Tensor] = None
    violated_label: Optional[int] = None
    score: Optional[float] = None