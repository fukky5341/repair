


from network_bound.bounder import IndividualBounds



def get_concrete_bounds(
        net, in_lb, in_ub,
        optimize_alpha=True,
        alpha_steps=20,
        alpha_lr=1e-1,
        save_coeffs=False,
        verbose=False
        ):
    """ returns concrete lower and upper bounds """

    bounder = IndividualBounds(
        net = net,
        lb_inp = in_lb,
        ub_inp = in_ub,
        device = in_lb.device,
    )

    return bounder.run(
        optimize_alpha=optimize_alpha,
        alpha_steps=alpha_steps,
        alpha_lr=alpha_lr,
        save_coeffs=save_coeffs,
        verbose=verbose
    )

def check_violation(net, lb, ub, spec):
    # approximate checking
    bounder = IndividualBounds(
        net = net,
        lb_inp = lb,
        ub_inp = ub,
        device = lb.device,
    )

    objs, _ = bounder.compute_dual_min_objective(C=spec.C)

    is_violated, _ = spec.check_violation(objs)

    return is_violated