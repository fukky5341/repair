from dataclasses import dataclass

import torch


@dataclass
class Spec:
    C: torch.Tensor  # (num_constraints, num_outputs)

    def check_violation(self, objs, tol=1e-6):
        where_violated = (objs < -tol).nonzero(as_tuple=True)[0]
        is_violated = len(where_violated) > 0
        return is_violated, where_violated
    
    def violation_loss(self, y):
        vals = self.C @ y
        return vals.min()