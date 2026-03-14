# REPAIR

## Tools
- [REGLO](https://github.com/BU-DEPEND-Lab/REGLO)
- [APRNN](https://github.com/95616ARG/APRNN)


## Datasets
- [MNIST C](https://zenodo.org/records/3239543)


## Pointwise Repair
1. get buggy inputs
    - this is done in APRNN
    - positive samples: $p = (p_1, p_2, ..., p_n)$
    - negative samples: $n = (n_1, n_2, ..., n_m)$
2. select layer to repair
3. build subspace for k-th layer
    - subspace is given from Dual formulation
    - $ss = ss_1 \cup ss_2 \cup ... \cup ss_k$ such that $p_1, p_2, ..., p_n$ are in the subspace and the final output of entire region in $ss$ satisfy the output specification
4. define the optimization problem
    - objective:
        - minimize the parameter change
        - minimize the distance between the negative samples $n$ and the subspace
    - constraints:
        - after modification, positive samples $p$ must be in the subspace
5. solve the optimization problem
6. if all negative samples are in the subspace, then stop; otherwise, iterate from step 2


## Regionwise Repair
1. collect repaired regions
    - origianl point is positive but, damaged input is negative
    - add perturbation on such input points to convert them to space
2. select layer to repair
3. build subspace for k-th layer
    - subspace is given from Dual formulation
    - $ss = ss_1 \cup ss_2 \cup ... \cup ss_k$ such that $P_1, P_2, ..., P_n$ are in the subspace and the final output of entire region in $ss$ satisfy the output specification
4. define the optimization problem
    - objective:
        - minimize the parameter change
        - minimize the distance between the negative regions $N$ and the subspace
    - constraints:
        - after modification, positive regions $P$ must be in the subspace
5. solve the optimization problem
6. if all negative regions are in the subspace, then stop; otherwise, iterate from step 2


## Plans
- subspace construction
- repair optimization
    - distance between negative samples and subspace
        - center point of the negative approximate region ($N[:k](n), n is negative sample)
        - distance: $||N[:k](n) - center(ss)||_2$
    - keep the positive samples in the subspace
        - symbolic approximate bound of $N[:k]$ with the shiftable parameters $\theta$ for the k-th layer
            $$
            N[:k]_{\theta}(p) \in [lb_{\theta)(p), ub_{\theta}(p)]
            $$
        - costraints:
            $$
            lb_{subsp} \leq lb_{\theta}(p) \leq ub_{\theta}(p) \leq ub_{subsp}
            $$
    - LP solver
        - objective: minimize the parameter change and the negatie distance
        - constraints: the same as above
    - barrier method
        - objective (gradient): minimize the parameter change and the negatie distance + barrier term for the constraints
        - constraints: the same as above


## How to get subspace
1. propagate the positive inputs to the k-th layer and get approximate lb and ub for $N[:k+1]$ (output of k-th layer).
2. compute gradient in the direction to violating the output constraints, such as $y_t - y_i < 0$
3. expand lb and ub by some margin to get the larger subspace.
    - by applying bounding method to the $lb' = lb - margin$ and $ub' = ub + margin$.
    - if the output specification is satisfied, try to further expand the subspace by increasing the margin.
    - if the output specification is not satisfied, try to shrink the subspace by decreasing the margin.

### expansion strategy
I the classification task, a violation function can be defined as $g(z) = y_t - y_i$, where $y_t$ is the output for the true class and $y_i$ is the output for the incorrect class. At the safe region, we have $g(z) > 0$, and at the violated region, we have $g(z) \leq 0$.
Then, we can use the first-order Taylor expansion to estimate the change in $g(z)$ when we expand the subspace by $\Delta z$:
$$
g(z + \Delta z) \approx g(z) + \nabla g(z)^T \Delta z
$$
If we want the largest safe step $t$:
$$
g(z) + \nabla g(z)^T (t d) = 0 \\
t = -\frac{g(z)}{\nabla g(z)^T d}
$$
This gives a first-order estimate of distance to violation boundary.
