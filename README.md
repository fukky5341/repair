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