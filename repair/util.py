from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Spec:
    '''
    specification is defined:
        c_i = y_t - y_i for i != t
        C = [c_1, c_2, ..., c_{t-1}, c_{t+1}, ..., c_n]  # shape: (num_constraints, num_outputs)
    '''
    C: torch.Tensor  # (num_constraints, num_outputs)
    target_label: Optional[int] = None

    def check_violation(self, objs, tol=1e-6):
        where_violated = (objs < -tol).nonzero(as_tuple=True)[0]
        is_violated = len(where_violated) > 0
        return is_violated, where_violated
    
    def violation_loss(self, y):
        vals = self.C @ y
        return vals.min()