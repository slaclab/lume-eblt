from __future__ import annotations

import logging
import os
import pathlib
import platform
import shlex
import shutil
import traceback
from time import monotonic
from typing import Any, ClassVar, Dict, Optional, Union

import h5py
import psutil
from lume import tools as lume_tools
from lume.base import CommandWrapper
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import pmd_unit
from typing_extensions import override

import tools
#from ..errors import EBLTRunFailure
#from . import parsers


from .input import EBLTInput, assign_names_to_elements, DriftTube, Bend, RFCavity,Wakefield
from .output import EBLTOutput
from .particles import EBLTParticleData

from .types import AnyPath

logger = logging.getLogger(__name__)


def find_mpirun():
    """
    Simple helper to find the mpi run command for macports and homebrew,
    as well as custom commands for Perlmutter at NERSC.
    """

    for p in [
        # Highest priority is what our PATH says:
        shutil.which("mpirun"),
        # Second, macports:
        "/opt/local/bin/mpirun",
        # Third, homebrew:
        "/opt/homebrew/bin/mpirun",
    ]:
        if p and os.path.exists(p):
            return f'"{p}"' + " -n {nproc} {command_mpi}"

    if os.environ.get("NERSC_HOST") == "perlmutter":
        srun = "srun -n {nproc} --ntasks-per-node {nproc} -c 1 {command_mpi}"
        hostname = platform.node()
        assert hostname  # This must exist
        if hostname.startswith("nid"):
            # Compute node
            return srun
        else:
            # This will work on a login node
            return "salloc -N {nnode} -C cpu -q interactive -t 04:00:00 " + srun

    # Default
    return "mpirun -n {nproc} {command_mpi}"


def find_workdir():
    if os.environ.get("NERSC_HOST") == "perlmutter":
        return os.environ.get("SCRATCH")
    else:
        return None


#def _make_eblt_input():
#    pass


class EBLT(CommandWrapper):
    """ """

    COMMAND: ClassVar[str] = "xeblt"
    COMMAND_MPI: ClassVar[str] = "xeblt-par"
    MPI_RUN: ClassVar[str] = find_mpirun()
    WORKDIR: ClassVar[Optional[str]] = find_workdir()

    # Environmental variables to search for executables
    command_env: str = "EBLT_BIN"
    command_mpi_env: str = "EBLT_BIN"
    original_path: AnyPath

    _input: EBLTInput
    output: Optional[EBLTOutput]
    initial_particles: Optional[EBLTParticleData] = None

    def __init__(
        self,
        input: Union[EBLTInput, str, pathlib.Path],
        *,
        workdir: Optional[Union[str, pathlib.Path]] = None,
        output: Optional[EBLTOutput] = None,
        command: Optional[str] = None,
        command_mpi: Optional[str] = None,
        use_mpi: bool = False,
        mpi_run: str = "",
        use_temp_dir: bool = True,
        verbose: bool = tools.global_display_options.verbose >= 1,
        timeout: Optional[float] = None,
        initial_particles: Optional[Union[ParticleGroup, EBLTParticleData]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            command=command,
            command_mpi=command_mpi,
            use_mpi=use_mpi,
            mpi_run=mpi_run,
            use_temp_dir=use_temp_dir,
            workdir=workdir,
            verbose=verbose,
            timeout=timeout,
            **kwargs,
        )
        self.original_path = workdir

        if not isinstance(input, EBLTInput):
            # We have either a string or a filename for our main input.
            self._input = EBLTInput.from_file(input)
            self.original_path, _ = os.path.split(input)
            assign_names_to_elements(self._input.lattice_lines)


        if initial_particles:
            if isinstance(initial_particles, ParticleGroup):
                self.initial_particles = EBLTParticleData.from_ParticleGroup(initial_particles)
            else:
                self.initial_particles = initial_particles


        if workdir is None:
            workdir = pathlib.Path(".")


        self._input = input
        self.output = output

        # MPI
        self.nproc = 1
        self.nnode = 1

    @property
    def input(self) -> EBLTInput:
        """EBLT input"""
        return self._input

    @input.setter
    def input(self, inp: Any) -> None:
        if not isinstance(inp, EBLTInput):
            raise ValueError(
                f"The provided input is of type {type(inp).__name__} and not `EBLTInput`. "
                f"Please consider creating a new EBLT object instead with the "
                f"new parameters!"
            )
        self._input = inp

    @property
    def nproc(self):
        """
        Number of MPI processes to use.
        """
        return self._nproc

    @nproc.setter
    def nproc(self, nproc: Optional[int]):
        if nproc is None or nproc == 0:
            nproc = psutil.cpu_count(logical=False)
        elif nproc < 0:
            nproc += psutil.cpu_count(logical=False)

        if nproc <= 0:
            raise ValueError(f"Calculated nproc is invalid: {nproc}")

        self._nproc = nproc

    @override
    def configure(self):
        """
        Configure and set up for run.
        """
        self.setup_workdir(self._workdir)
        self.vprint("Configured to run in:", self.path)
        self.configured = True
        self.finished = False

    @override
    def run(
        self,
        load_particles: bool = False,
        raise_on_error: bool = True,
    ) -> EBLTOutput:
        """
        Execute EBLT with the configured input settings.

        Parameters
        ----------
        load_particles : bool, default=False
            After execution, load all particle files.
        raise_on_error : bool, default=True
            If EBLT fails to run, raise an error. Depending on the error,
            output information may still be accessible in the ``.output``
            attribute.

        Returns
        -------
        EBLTOutput
            The output data.  This is also accessible as ``.output``.
        """
        if not self.configured:
            self.configure()

        if self.path is None:
            raise ValueError("Path (base_path) not yet set")

        self.finished = False

        runscript = self.get_run_script()

        start_time = monotonic()
        self.vprint(f"Running EBLT in {self.path}")
        self.vprint(runscript)

        self.write_input()

        if self.timeout:
            self.vprint(
                f"Timeout of {self.timeout} is being used; output will be "
                f"displaye after EBLT exits."
            )
            execute_result = tools.execute2(
                shlex.split(runscript),
                timeout=self.timeout,
                cwd=self.path,
            )
            self.vprint(execute_result["log"])
        else:
            log = []
            try:
                for line in tools.execute(shlex.split(runscript), cwd=self.path):
                    self.vprint(line, end="")
                    log.append(line)
            except Exception as ex:
                log.append(f"EBLT exited with an error: {ex}")
                self.vprint(log[-1])
                execute_result = {
                    "log": "".join(log),
                    "error": True,
                    "why_error": "error",
                }
            else:
                execute_result = {
                    "log": "".join(log),
                    "error": False,
                    "why_error": "",
                }

        end_time = monotonic()

        self.finished = True
        run_info = RunInfo(
            run_script=runscript,
            error=execute_result["error"],
            error_reason=execute_result["why_error"],
            start_time=start_time,
            end_time=end_time,
            run_time=end_time - start_time,
            output_log=execute_result["log"],
        )

        success_or_failure = "Success" if not execute_result["error"] else "Failure"
        self.vprint(f"{success_or_failure} - execution took {run_info.run_time:0.2f}s.")

        try:
            self.output = self.load_output()

        except Exception as ex:
            stack = traceback.format_exc()
            run_info.error = True
            run_info.error_reason = (
                f"Failed to load output file. {ex.__class__.__name__}: {ex}\n{stack}"
            )
            self.output = EBLTOutput(run=run_info)
            if hasattr(ex, "add_note"):
                # Python 3.11+
                ex.add_note(
                    f"\nEBLT output was:\n\n{execute_result['log']}\n(End of EBLT output)"
                )
            if raise_on_error:
                raise

        self.output.run = run_info
        if run_info.error and raise_on_error:
            raise EBLTRunFailure(f"EBLT failed to run: {run_info.error_reason}")

        return self.output

    def get_executable(self):
        """
        Gets the full path of the executable from .command, .command_mpi
        Will search environmental variables:
                Genesis4.command_env='GENESIS4_BIN'
                Genesis4.command_mpi_env='GENESIS4_BIN'
        """
        if self.use_mpi:
            return lume_tools.find_executable(
                exename=self.command_mpi, envname=self.command_mpi_env
            )
        return lume_tools.find_executable(
            exename=self.command, envname=self.command_env
        )

    def get_run_prefix(self) -> str:
        """Get the command prefix to run Genesis (e.g., 'mpirun' or 'genesis4')."""
        exe = self.get_executable()

        if self.nproc != 1 and not self.use_mpi:
            self.vprint(f"Setting use_mpi = True because nproc = {self.nproc}")
            self.use_mpi = True

        if self.use_mpi:
            return self.mpi_run.format(
                nnode=self.nnode, nproc=self.nproc, command_mpi=exe
            )
        return exe

    @override
    def get_run_script(self, write_to_path: bool = True) -> str:
        """
        Assembles the run script using self.mpi_run string of the form:
            'mpirun -n {n} {command_mpi}'
        Optionally writes a file 'run' with this line to path.

        mpi_exe could be a complicated string like:
            'srun -N 1 --cpu_bind=cores {n} {command_mpi}'
            or
            'mpirun -n {n} {command_mpi}'
        """
        if self.path is None:
            raise ValueError("path (base_path) not yet set")

        runscript = shlex.join(
            [
                *shlex.split(self.get_run_prefix()),
                *self.input.arguments,
            ]
        )

        if write_to_path:
            self.write_run_script()

        return runscript

    def write_run_script(self, path: Optional[AnyPath] = None) -> pathlib.Path:
        """
        Write the 'run' script which can be used in a terminal to launch Genesis.

        This is also performed automatically in `write_input` and
        `get_run_script`.

        Parameters
        -------
        path : pathlib.Path or str
            Where to write the run script.  Defaults to `{self.path}/run`.

        Returns
        -------
        pathlib.Path
            The run script location.
        """
        path = path or self.path
        if path is None:
            raise ValueError("path (base_path) not yet set and no path specified")

        path = pathlib.Path(path)
        if path.is_dir():
            path = path / "run"

        self.input.write_run_script(
            path,
            command_prefix=self.get_run_prefix(),
        )
        logger.debug("Wrote run script to %s", path)
        return path

    @override
    def write_input(
        self,
        path: Optional[AnyPath] = None,
        write_run_script: bool = True,
    ):
        """
        Write the input parameters into the file.

        Parameters
        ----------
        path : str, optional
            The directory to write the input parameters
        """
        if not self.configured:
            self.configure()

        if path is None:
            path = self.path

        if path is None:
            raise ValueError("Path has not yet been set; cannot write input.")

        path = pathlib.Path(path)

        # write wakefield
        self.write_wakefield(path)

        # write initial particles
        self.write_initial_particles(path)



        # write main input file
        filename = os.path.join(path, 'eblt.in')
        self.input.to_file(filename=filename)



        # write run script
        if write_run_script:
            self.write_run_script(path)

    def write_initial_particles(self, path: Optional[AnyPath] = None) -> None:
        if self.initial_particles:
            self.initial_particles.write_EBLT_input(path)
            #update header
            self._input.parameters.flagdist = 100
            # update beam radius
            if self.initial_particles.beam_radius:
                self.update_beam_radius(self.initial_particles.beam_radius)

        elif self._input.parameters.flagdist in [100, 200, 300]:
            src = os.path.join(self.original_path, 'pts.in')
            dest = os.path.join(path, 'pts.in')

            assert os.path.isfile(src), "Initial particles file not found"

            # Don't worry about overwriting in temporary directories
            if self._tempdir and os.path.exists(dest):
                os.remove(dest)

            if not os.path.exists(dest):
                shutil.copyfile(src, dest)
                #writers.write_input_particles_from_file(src, dest, self.header['Np'])
            else:
                self.vprint('pts.in already exits, will not overwrite.')

    def update_beam_radius(self, r: float) -> None:
        print('Updating beam radius in the lattice to be ', r)
        for lattice_element in self._input.lattice_lines:
            if (isinstance(lattice_element, DriftTube) or
                    isinstance(lattice_element, Bend) or
                    isinstance(lattice_element, RFCavity)):
                lattice_element.V1 = r

    def write_wakefield(self, path: Optional[AnyPath] = None) -> None:
        rec = []
        for lattice_element in self._input.lattice_lines:
            if isinstance(lattice_element, Wakefield):
                file_id = lattice_element.file_id

                if file_id in rec:
                    continue

                filename = 'rfdata' + str(file_id)
                src = os.path.join(self.original_path, filename)

                assert os.path.isfile(src), "Wakefield file not found"

                dest = os.path.join(path, filename)

                # Don't worry about overwriting in temporary directories
                if self._tempdir and os.path.exists(dest):
                    os.remove(dest)

                if not os.path.exists(dest):
                    shutil.copyfile(src, dest)
                    # writers.write_input_particles_from_file(src, dest, self.header['Np'])
                else:
                    self.vprint(f'{} already exits, will not overwrite.'.format(filename))

                rec.append(file_id)

    @override
    def load_output(self) -> EBLTOutput:
        return EBLTOutput.from_directory(self.path)


    def plot(self):
        pass

    def stat(self):
        pass

    @override
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, EBLT):
            return False
        return self.input == other.input and self.output == other.output

    @override
    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, EBLT):
            return False
        return self.input != other.input or self.output != other.output
