from dataclasses import dataclass, field
import torch
from typing import Optional

@dataclass
class Region:
    lb: torch.Tensor          # shape: (1, C, H, W) or (1, D)
    ub: torch.Tensor          # same shape as lb
    target_label: int
    depth: int = 0
    region_id: Optional[int] = None
    parent_id: Optional[int] = None

    # filled after analysis
    status: str = "unprocessed"   # "positive", "negative", "undecided"
    margin_lbs: Optional[torch.Tensor] = None   # shape: (num_classes-1,)
    candidate_x: Optional[torch.Tensor] = None
    violated_label: Optional[int] = None
    score: Optional[float] = None