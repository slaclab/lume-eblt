from distgen import Generator
import os
from .run import EBLT
def run_impact_with_distgen(settings=None,
                            distgen_input_file=None,
                            impact_config=None,
                            workdir=None,
                            verbose=False):
    """
    Creates, runs, and returns an Impact object using distgen input.

    .distgen_input = parsed distgen.Generatator's .input is attached to the object.

    """

    # setup objects
    if isinstance(impact_config, str):
        I = Impact.from_yaml(impact_config)
    else:
        I = Impact(**impact_config)

    if workdir:
        I._workdir = workdir  # TODO: fix in LUME-Base
        I.configure()  # again

    I.verbose = verbose
    G = Generator(distgen_input_file)
    G.verbose = verbose

    if settings:
        for key in settings:
            val = settings[key]
            if key.startswith('distgen:'):
                key = key[len('distgen:'):]
                if verbose:
                    print(f'Setting distgen {key} = {val}')
                G[key] = val
            else:
                # Assume impact
                if verbose:
                    print(f'Setting impact {key} = {val}')
                I[key] = val

                # Get particles
    G.run()
    P = G.particles

    # Attach particles
    I.initial_particles = P

    # Attach distgen input. This is non-standard.
    I.distgen_input = G.input

    I.run()

    return I