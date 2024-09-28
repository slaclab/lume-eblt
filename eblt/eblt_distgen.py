from distgen import Generator
import os


from .run import EBLT
from h5py import File
from .evaluate import  default_eblt_merit
from . import tools
from typing import Optional, Dict, Union, Callable
from .types import AnyPath
import distgen

def run_eblt_with_distgen(eblt_config: Union[str, Dict],
                          settings: Optional[Dict] = None,
                          distgen_input_file: Optional[str]=None,
                          workdir: Optional[AnyPath] = None,
                          verbose: bool = False):
    """
    Creates, runs, and returns an Impact object using distgen input.

    .distgen_input = parsed distgen.Generatator's .input is attached to the object.

    """

    # setup objects
    if isinstance(eblt_config, str):
        I = EBLT.from_yaml(eblt_config)
    else:
        I = EBLT(**eblt_config)

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
                # Assume EBLT
                if verbose:
                    print(f'Setting impact {key} = {val}')
                if key.startwith('parameters:'):
                    key = key[len('parameters:'):]
                    setattr(I.input.parameters, key, val)
                elif key == 'phase_space_coefficients':
                    I.input.phase_space_coefficients.coefficients = val
                elif key == 'current_coefficients':
                    I.input.phase_space_coefficients.coefficients = val
                else:
                    # Assume lattice
                    key = key.split(':')
                    for element in I.input.lattice_lines:
                        if element.name == key[0]:
                            setattr(element, key[1], val)

    # Get particles
    G.run()
    P = G.particles

    # Attach particles
    I.initial_particles = P

    I.run()

    return I




def evaluate_eblt_with_distgen(eblt_config: Union[str, Dict],
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

    I = run_eblt_with_distgen(settings=settings,
                                distgen_input_file=distgen_input_file,
                                eblt_config=eblt_config,
                                workdir=workdir,
                                verbose=verbose)

    if merit_f:
        output = merit_f(I)
    else:
        output = default_eblt_merit(I)

    if 'error' in output and output['error']:
        raise ValueError('run_impact_with_distgen returned error in output')

    # Recreate Generator object for fingerprint, proper archiving

    G = Generator(distgen_input_file)

    fingerprint = fingerprint_eblt_with_distgen(I, G)
    output['fingerprint'] = fingerprint

    if archive_path:
        path = tools.full_path(archive_path)
        assert os.path.exists(path), f'archive path does not exist: {path}'
        archive_file = os.path.join(path, fingerprint + '.h5')
        output['archive'] = archive_file

        # Call the composite archive method
        archive_eblt_with_distgen(I, G, archive_file=archive_file)

    return output


def fingerprint_eblt_with_distgen(eblt_object: EBLT, distgen_object: distgen):
    """
    Calls fingerprint() of each of these objects
    """
    f1 = eblt_object.fingerprint()
    f2 = distgen_object.fingerprint()
    d = {'f1': f1, 'f2': f2}
    return tools.fingerprint(d)


def archive_eblt_with_distgen(eblt_object,
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
    eblt_object.archive(g)

    h5.close()