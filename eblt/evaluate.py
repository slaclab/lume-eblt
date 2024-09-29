
from .run import EBLT

def default_eblt_merit(E: EBLT):
    """
    merit function to operate on an evaluated LUME-EBLT object E.

    Returns dict of scalar values
    """
    # Check for error
    if E.output.run.error:
        return {'error': True}
    else:
        m = {'error': False}

    # Gather stat output
    stats_keys = ['kinetic_energy', 'gamma', 'mean_z', 'rms_z' , 'mean_delta_gamma', 'rms_delta_gamma']
    for k in stats_keys:
        m['end_' + k] = getattr(E.output.stats, k)[-1]

    m['run_time'] = E.output.run.run_time

    P = E.output.particle_distributions[201].to_particlegroup()
    P_init = E.output.particle_distributions[101].to_particlegroup()

    # All impact particles read back have status==1
    #
    ntotal = len(P_init)
    nlost = ntotal - len(P)

    m['end_n_particle_loss'] = nlost

    # Get live only for stat calcs
    P = P.where(P.status == 1)

    # No live particles
    if len(P) == 0:
        return {'error': True}

    # Special
    m['end_total_charge'] = P['charge']
    m['end_higher_order_energy_spread'] = P['higher_order_energy_spread']


    # Remove annoying strings
    if 'why_error' in m:
        m.pop('why_error')

    return m