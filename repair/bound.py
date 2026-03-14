


from network_bound.bounder import IndividualBounds



def get_concrete_bounds(
        net, in_lb, in_ub,
        optimize_alpha=True,
        alpha_step=20,
        alpha_lr=1e-1,
        save_coeffs=False,
        verbose=False
        ):
    """ returns concrete lower and upper bounds """

    bounder = IndividualBounds(
        net = net,
        in_lb = in_lb,
        in_ub = in_ub,
        device = in_lb.device,
    )

    return bounder.run(
        optimize_alpha=optimize_alpha,
        alpha_step=alpha_step,
        alpha_lr=alpha_lr,
        save_coeffs=save_coeffs,
        verbose=verbose
    )

def check_violation(netS, lb, ub, spec):
    # approximate checking
    bounder = IndividualBounds(
        net = netS,
        in_lb = lb,
        in_ub = ub,
        device = lb.device,
    )

    objs, _ = bounder.compute_dual_min_objective(C=spec.C)

    is_violated, _ = spec.check_violation(objs)

    return is_violated