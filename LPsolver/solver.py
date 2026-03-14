from dataclasses import dataclass, field
from enum import Enum
import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB



class ObjectiveType(Enum):
    SLACK_INTERVAL = "slack_interval"
    HINGE_MARGIN = "hinge_margin"
    PARAM_CHANGE = "param_change"
    WEIGHTED_SLACK = "weighted_slack"


@dataclass
class LPSolver:

    weight: torch.Tensor
    bias: torch.Tensor
    modifiable_range: float = 1.0

    model: gp.Model = field(init=False)
    dw: dict = field(init=False)
    db: dict = field(init=False)

    objective_terms: list = field(default_factory=list)

    n_out: int = field(init=False)
    n_in: int = field(init=False)

    def __post_init__(self):
        self.W = self.weight.detach().cpu().numpy()
        self.bias = self.bias.detach().cpu().numpy()
        W = self.W
        b = self.bias

        self.n_out, self.n_in = W.shape
        self.model = gp.Model("repair_lp")
        self.model.setParam("OutputFlag", 0)
        self.dw = {}
        self.db = {}

        for i in range(self.n_out):
            for j in range(self.n_in):
                self.dw[i,j] = self.model.addVar(
                    lb=-self.modifiable_range,
                    ub=self.modifiable_range,
                    name=f"dw_{i}_{j}"
                )

            self.db[i] = self.model.addVar(
                lb=-self.modifiable_range,
                ub=self.modifiable_range,
                name=f"db_{i}"
            )

        self.model.update()


    def add_sign_constraints(self):
        W = self.weight.detach().cpu().numpy()

        for i in range(self.n_out):
            for j in range(self.n_in):
                w = W[i,j]
                var = self.dw[i,j]

                if w > 0:
                    self.model.addConstr(var >= -w)
                elif w < 0:
                    self.model.addConstr(var <= -w)


    def add_region(self, z_lb, z_ub, C_lb, C_ub):
        W = self.W
        b = self.bias

        for i in range(self.n_out):
            expr_lb = gp.LinExpr()
            expr_ub = gp.LinExpr()

            for j in range(self.n_in):
                w = W[i,j]
                dw = self.dw[i,j]
                if w >= 0:
                    alpha = z_lb[j]
                    beta = z_ub[j]
                else:
                    alpha = z_ub[j]
                    beta = z_lb[j]
                expr_lb += alpha * (w + dw)
                expr_ub += beta * (w + dw)

            expr_lb += b[i] + self.db[i]
            expr_ub += b[i] + self.db[i]
            eps = 1e-9
            self.model.addConstr(expr_lb >= C_lb[i] - eps)
            self.model.addConstr(expr_ub <= C_ub[i] + eps)


    # --- objective ---
    def compute_region_weight(self, n_region, subspace):
        """
        Compute importance weight for a repaired region based on
        how far its current output is from the safe subspace.
        """

        violation = 0.0

        for i in range(self.n_out):

            y = n_region.center_output[i]

            if y < subspace.lb[i]:
                violation += subspace.lb[i] - y

            elif y > subspace.ub[i]:
                violation += y - subspace.ub[i]

        return 1.0 + violation
    
    def _objective_slack_interval(self, expr, lb, ub):
        s_low = self.model.addVar(lb=0)
        s_up = self.model.addVar(lb=0)

        self.model.addConstr(expr + s_low >= lb)
        self.model.addConstr(expr - s_up <= ub)

        return s_low + s_up
    
    def _objective_hinge(self, expr, lb):
        slack = self.model.addVar(lb=0)
        self.model.addConstr(expr + slack >= lb)

        return slack
    
    def _objective_param_change(self):
        return gp.quicksum(
            self.dw[i,j]*self.dw[i,j]
            for i in range(self.n_out)
            for j in range(self.n_in)
        )
    
    def _objective_weighted_slack(self, expr, lb, ub, weight):

        s_low = self.model.addVar(lb=0)
        s_up = self.model.addVar(lb=0)

        self.model.addConstr(expr + s_low >= lb)
        self.model.addConstr(expr - s_up <= ub)

        return weight * (s_low + s_up)

    def set_objective(self, n_region, subspace, obj_type):

        z_lb = n_region.z_lb
        z_ub = n_region.z_ub

        W = self.W
        b = self.bias

        terms = []
        for i in range(self.n_out):
            expr = gp.LinExpr()
            for j in range(self.n_in):
                w = W[i,j]
                dw = self.dw[i,j]
                z = z_lb[j] if w >= 0 else z_ub[j]
                expr += z * (w + dw)
            expr += b[i] + self.db[i]

            if obj_type == ObjectiveType.SLACK_INTERVAL:
                terms.append(
                    self._objective_slack_interval(
                        expr,
                        subspace.lb[i],
                        subspace.ub[i]
                    )
                )

            elif obj_type == ObjectiveType.HINGE_MARGIN:
                terms.append(
                    self._objective_hinge(
                        expr,
                        subspace.lb[i]
                    )
                )

            elif obj_type == ObjectiveType.WEIGHTED_SLACK:
                weight = self.compute_region_weight(n_region, subspace)
                terms.append(
                    self._objective_weighted_slack(
                        expr,
                        subspace.lb[i],
                        subspace.ub[i],
                        weight
                    )
                )

        self.objective_terms.extend(terms)

    # add a regularization term to encourage smaller parameter changes
    # and collected objective terms into a single objective function
    def build_objective(self):
        param_reg = gp.quicksum(
            self.dw[i,j]*self.dw[i,j]
            for i in range(self.n_out)
            for j in range(self.n_in)
        )
        self.model.addConstr(param_reg <= 10)

        obj = gp.quicksum(self.objective_terms) + 1e-4 * param_reg

        self.model.setObjective(obj, GRB.MINIMIZE)


# --- solve ---
    def solve(self):
        self.model.optimize()
        if self.model.status != GRB.OPTIMAL:
            return None

        W = self.W
        B = self.bias
        new_W = W.copy()
        new_B = B.copy()

        for i in range(self.n_out):
            for j in range(self.n_in):
                new_W[i,j] += self.dw[i,j].X

        for i in range(self.n_out):
            new_B[i] += self.db[i].X

        return (
            torch.tensor(new_W),
            torch.tensor(new_B)
        )