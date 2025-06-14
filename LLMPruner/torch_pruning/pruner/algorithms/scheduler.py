def linear_scheduler(ch_sparsity, steps):
    return [((i) / float(steps)) * ch_sparsity for i in range(steps+1)], ch_sparsity / steps