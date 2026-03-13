# REGLO code analysis

[repair entrance link](https://github.com/BU-DEPEND-Lab/REGLO/blob/a78dbbe119d49a8934f40b37658d02a8cce5f492/net_repair/reglo.py#L124)



## Pipeline of Repair

1. compute symbolic bounds from input layer to the previous layer of the final layer
    $$
    \Delta z^{(L-1)} \in [A_{lb} x + b_{lb},\; A_{ub} x + b_{ub}]
    $$
2. add constraints so that the violating samples satisfy the output specification
3. solve the optimization problem via Barrier method

This repair process is performed in REGLO/net_repair/reglo_mnist.py, specifically in the function `repair_one_step` of REGLO/net_repair/reglo.py.



## repair_one_step (main function for repair)

### Step 1: collect constraints for violated samples
```
for sample in violated_samples:
    dA_lb, dA_ub, bias_lb, bias_ub = self.last_hidden_layer_bounds(sample)
    repair_engine.add_constraints([dA_lb, dA_ub, bias_lb, bias_ub])
    self.repair_regions.append(sample)
```
for each violated sample:
1. `last_hidden_bounds`: compute the symbolic bounds through the last hidden layer
2. `add_constraints`: add the constraints
    $$
    \Delta z^{(L)} = W'^{(L)} \Delta z^{(L-1)} \\
    \therefore \text{constraint} = \{ W'^{(L)} (A_{lb} x + b_{lb})  - k \leq 0 \}
    $$
    where $k$ is the output specification threshold


### Step 2: solve the optimization problem (via Barrier method)

This step is performed for updating the parameters of the final layer. 
```
delta_theta = repair_engine.repair(repair_steps, step_size=repair_step_size, t=t*areas_num).T
```

In the function `repair` of REGLO/net_repair/repair_core.py ([link](https://github.com/BU-DEPEND-Lab/REGLO/blob/a78dbbe119d49a8934f40b37658d02a8cce5f492/net_repair/repair_core.py#L39)), the optimization problem is solved via Barrier method.

Specifically, the barrier method is implemented in the function `update`. Optimization loop:
```
while self.curr_step < self.step_num:
    self.is_retreat = False
    self.update()
```
This updates $W' = \theta + \Delta \theta$ iteratively.


1. Initialization: $ \Delta \theta = - \theta $
    $$
    \because W'^{(L)} = \theta + \Delta \theta \rightarrow \Delta z^{(L)} = 0 \le k \quad (\text{specification threshold})
    $$
2. Gradient computation: 
    $$
    (\text{parameter term}): \quad \nabla_{param} = \frac{\Delta \theta}{\|\Delta \theta\|_p} \\
    (\text{constraint/barrier term}): \quad = \sum_{i} - \frac{1}{t} \frac{1}{\epsilon - g_i(\theta)} \frac{\partial g_i}{\partial \theta} \\
    \therefore \nabla = \nabla_{param} + \nabla_{constraint}
    $$
    - if violated constraints are detected, `retreat` is applied:
        - to set ` self.delta-theta = self.prev_delta_theta`
        - to increase barrier parameter $t$
        - to set `is_retreat = True`
3. Update: $ \Delta \theta = \Delta \theta - \alpha * \nabla $ where $\alpha$ is the step size
4. Update barrier parameter if there still exist violating constraints: $t = t * \beta_1$ where $\beta_1$ is the hyperparameter for updating the barrier parameter
5. Update step size: $\alpha = \alpha * \beta_0$ where $\beta_0$ is the hyperparameter for updating the step size
6. Iterate from step 2 until the maximum number of steps is reached
