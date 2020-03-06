import numpy as np

def round_up(number):
    rounded = int(number)
    if rounded == number:
        return rounded
    else:
        return rounded + 1

def get_effective_point_number(time, time_step):
    '''
    function for discretization of time depending on the sample rate of the AWG.
    Args:
        time (double): time in ns of which you want to know how many points the AWG needs to get there
        time_step (double) : time step of the AWG (ns)

    Returns:
        how many points you need to get to the desired time step.
    '''
    n_pt, mod = divmod(time, time_step)
    if mod > time_step/2:
        n_pt += 1

    return int(n_pt)


