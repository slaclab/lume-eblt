from distgen import Generator
import os


from .run import BELT
from h5py import File
from .evaluate import  default_belt_merit
from . import tools
from typing import Optional, Dict, Union, Callable
from .types import AnyPath
import distgen

def run_belt_with_distgen(belt_config: Union[str, Dict],
                          settings: Optional[Dict] = None,
                          distgen_input_file: Optional[str]=None,
                          workdir: Optional[AnyPath] = None,
                          verbose: bool = False):
    """
    Creates, runs, and returns an Impact object using distgen input.

    .distgen_input = parsed distgen.Generatator's .input is attached to the object.

    """

    # setup objects
    if isinstance(belt_config, str):
        E = BELT.from_yaml(belt_config)
    else:
        E = BELT(**belt_config)

    if workdir:
        E._workdir = workdir  # TODO: fix in LUME-Base
        E.configure()  # again

    E.verbose = verbose
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
                # Assume BELT
                if verbose:
                    print(f'Setting impact {key} = {val}')
                if key.startswith('parameters:'):
                    key = key[len('parameters:'):]
                    setattr(E.input.parameters, key, val)
                elif key == 'phase_space_coefficients':
                    E.input.phase_space_coefficients.coefficients = val
                elif key == 'current_coefficients':
                    E.input.phase_space_coefficients.coefficients = val
                else:
                    # Assume lattice
                    key = key.split(':')
                    for element in E.input.lattice_lines:
                        if element.name == key[0]:
                            setattr(element, key[1], val)

    # Get particles
    G.run()
    P = G.particles

    # Attach particles
    E.initial_particles = P

    E.run()

    return E




def evaluate_belt_with_distgen(belt_config: Union[str, Dict],
                               settings: Optional[Dict],
                               distgen_input_file: Optional[str]=None,
                               workdir: Optional[AnyPath] = None,
                                 archive_path: Optional[AnyPath] = None,
                                 merit_f: Optional[Callable] = None,
                                 verbose: bool = False):
    """

    Similar to run_impact_with_distgen, but requires settings a the only positional argument.

    If an archive_path is given, the complete evaluated Impact and Generator objects will be archived
    to a file named using a fingerprint from both objects.

    If merit_f is given, this function will be applied to the evaluated Impact object, and this will be returned.

    Otherwise, a default function will be applied.


    """

    E = run_belt_with_distgen(settings=settings,
                                distgen_input_file=distgen_input_file,
                                belt_config=belt_config,
                                workdir=workdir,
                                verbose=verbose)

    if merit_f:
        output = merit_f(E)
    else:
        output = default_belt_merit(E)

    if 'error' in output and output['error']:
        raise ValueError('run_impact_with_distgen returned error in output')

    # Recreate Generator object for fingerprint, proper archiving

    G = Generator(distgen_input_file)

    fingerprint = fingerprint_belt_with_distgen(E, G)
    output['fingerprint'] = fingerprint

    if archive_path:
        path = tools.full_path(archive_path)
        assert os.path.exists(path), f'archive path does not exist: {path}'
        archive_file = os.path.join(path, fingerprint + '.h5')
        output['archive'] = archive_file

        # Call the composite archive method
        archive_belt_with_distgen(E, G, archive_file=archive_file)

    return output


def fingerprint_belt_with_distgen(belt_object: BELT, distgen_object: distgen):
    """
    Calls fingerprint() of each of these objects
    """
    f1 = belt_object.fingerprint()
    f2 = distgen_object.fingerprint()
    d = {'f1': f1, 'f2': f2}
    return tools.fingerprint(d)


def archive_belt_with_distgen(belt_object,
                                distgen_object,
                                archive_file=None,
                                impact_group='impact',
                                distgen_group='distgen'):
    """
    Creates a new archive_file (hdf5) with groups for
    impact and distgen.

    Calls .archive method of Impact and Distgen objects, into these groups.
    """

    h5 = File(archive_file, 'w')

    # fingerprint = tools.fingerprint(astra_object.input.update(distgen.input))

    g = h5.create_group(distgen_group)
    distgen_object.archive(g)

    g = h5.create_group(impact_group)
    belt_object.archive(g)

    h5.close()