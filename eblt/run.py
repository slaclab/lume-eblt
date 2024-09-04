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

from .. import tools
from ..errors import EbltRunFailure
from . import parsers
from .field import FieldFile
from .input import EbltInput, Lattice, MainInput
from .output import EbltOutput, RunInfo
from .particles import EbltParticleData
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

def _make_eblt_input():
    pass

class EBLT(CommandWrapper):
    """

    """
    COMMAND: ClassVar[str] = "xeblt"
    COMMAND_MPI: ClassVar[str] = "xeblt-par"
    MPI_RUN: ClassVar[str] = find_mpirun()
    WORKDIR: ClassVar[Optional[str]] = find_workdir()

    # Environmental variables to search for executables
    command_env: str = "EBLT_BIN"
    command_mpi_env: str = "EBLT_BIN"
    original_path: AnyPath

    _input: EbltInput
    output: Optional[EbltOutput]

    def __init__(
        self,
        input: Optional[Union[MainInput, EbltInput, str, pathlib.Path]] = None,
        lattice: Union[Lattice, str, pathlib.Path] = "",
        *,
        workdir: Optional[Union[str, pathlib.Path]] = None,
        output: Optional[EbltOutput] = None,
        alias: Optional[Dict[str, str]] = None,
        units: Optional[Dict[str, pmd_unit]] = None,
        command: Optional[str] = None,
        command_mpi: Optional[str] = None,
        use_mpi: bool = False,
        mpi_run: str = "",
        use_temp_dir: bool = True,
        verbose: bool = tools.global_display_options.verbose >= 1,
        timeout: Optional[float] = None,
        initial_particles: Optional[Union[ParticleGroup, EbltParticleData]] = None,
        initial_field: Optional[FieldFile] = None,
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

        if input is None:
            input = EbltInput(
                main=MainInput(),
                lattice=Lattice(),
                initial_particles=initial_particles,
            )
        elif isinstance(input, MainInput):
            input = EbltInput.from_main_input(
                main=input,
                lattice=lattice,
                source_path=pathlib.Path(workdir or "."),
            )
        elif not isinstance(input, EbltInput):
            # We have either a string or a filename for our main input.
            workdir, input = _make_eblt_input(
                input,
                lattice,
                source_path=workdir,
            )

        if (
                input.initial_particles is not initial_particles
                and initial_particles is not None
        ):
            input.initial_particles = initial_particles

        if input.initial_field is not initial_field and initial_field is not None:
            input.initial_field = initial_field

        if workdir is None:
            workdir = pathlib.Path(".")

        self.original_path = workdir
        self._input = input
        self.output = output

        # Internal
        self._units = dict(units or parsers.known_unit)
        self._alias = dict(alias or {})

        # MPI
        self.nproc = 1
        self.nnode = 1

    @property
    def input(self) -> EbltInput:
        """Eblt input"""
        return self._input

    @input.setter
    def input(self, inp: Any) -> None:
        if not isinstance(inp, EbltInput):
            raise ValueError(
                f"The provided input is of type {type(inp).__name__} and not `EbltInput`. "
                f"Please consider creating a new Eblt object instead with the "
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
    def run(self,
            load_particles: bool = False,
            raise_on_error: bool = True,
    ) -> EbltOutput:
        """
        Execute Eblt 4 with the configured input settings.

        Parameters
        ----------
        load_fields : bool, default=False
            After execution, load all field files.
        load_particles : bool, default=False
            After execution, load all particle files.
        smear : bool, default=True
            If set, for particles, this will smear the phase over the sample
            (skipped) slices, preserving the modulus.
        raise_on_error : bool, default=True
            If Eblt 4 fails to run, raise an error. Depending on the error,
            output information may still be accessible in the ``.output``
            attribute.

        Returns
        -------
        EbltOutput
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
            self.output = self.load_output(
                load_fields=load_fields,
                load_particles=load_particles,
                smear=smear,
            )
        except Exception as ex:
            stack = traceback.format_exc()
            run_info.error = True
            run_info.error_reason = (
                f"Failed to load output file. {ex.__class__.__name__}: {ex}\n{stack}"
            )
            self.output = EbltOutput(run=run_info)
            if hasattr(ex, "add_note"):
                # Python 3.11+
                ex.add_note(
                    f"\nEBLT output was:\n\n{execute_result['log']}\n(End of Eblt output)"
                )
            if raise_on_error:
                raise

        self.output.run = run_info
        if run_info.error and raise_on_error:
            raise EbltRunFailure(
                f"EBLT failed to run: {run_info.error_reason}"
            )

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
        self.input.write(workdir=path)
        if write_run_script:
            self.write_run_script(path)

    @property
    @override
    def initial_particles(self) -> Optional[Union[ParticleGroup, EbltParticleData]]:
        """Initial particles, if defined.  Property is alias for `.input.main.initial_particles`."""
        return self.input.initial_particles

    @initial_particles.setter
    def initial_particles(
            self,
            value: Optional[Union[ParticleGroup, EbltParticleData]],
    ) -> None:
        self.input.initial_particles = value

    def _archive(self, h5: h5py.Group):
        self.input.archive(h5.create_group("input"))
        if self.output is not None:
            self.output.archive(h5.create_group("output"))

    @override
    def archive(self, dest: Union[AnyPath, h5py.Group]) -> None:
        """
        Archive the latest run, input and output, to a single HDF5 file.

        Parameters
        ----------
        dest : filename or h5py.Group
        """
        if isinstance(dest, (str, pathlib.Path)):
            with h5py.File(dest, "w") as fp:
                self._archive(fp)
        elif isinstance(dest, (h5py.File, h5py.Group)):
            self._archive(dest)

    to_hdf5 = archive

    def _load_archive(self, h5: h5py.Group):
        self.input = Genesis4Input.from_archive(h5["input"])
        if "output" in h5:
            self.output = Genesis4Output.from_archive(h5["output"])
        else:
            self.output = None

    @override
    def load_archive(self, arch: Union[AnyPath, h5py.Group]) -> None:
        """
        Load an archive from a single HDF5 file into this Genesis4 object.

        Parameters
        ----------
        arch : filename or h5py.Group
        """
        if isinstance(arch, (str, pathlib.Path)):
            with h5py.File(arch, "r") as fp:
                self._load_archive(fp)
        elif isinstance(arch, (h5py.File, h5py.Group)):
            self._load_archive(arch)

    @override
    @classmethod
    def from_archive(cls, arch: Union[AnyPath, h5py.Group]) -> Genesis4:
        """
        Create a new Genesis4 object from an archive file.

        Parameters
        ----------
        arch : filename or h5py.Group
        """
        inst = cls()
        inst.load_archive(arch)
        return inst

    @override
    def load_output(self) -> EbltOutput:
        pass

    def plot(self):
        pass

    def stat(self):
        pass

    @override
    @staticmethod
    def input_parser(path: AnyPath) -> MainInput:
        """
        Invoke the specialized main input parser and returns the `MainInput`
        instance.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the main input file.

        Returns
        -------
        MainInput
        """
        return MainInput.from_file(path)

    @staticmethod
    def lattice_parser(path: AnyPath) -> Lattice:
        """
        Invoke the specialized lattice input parser and returns the `Lattice`
        instance.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the lattice input file.

        Returns
        -------
        Lattice
        """
        return Lattice.from_file(path)

    @override
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Eblt):
            return False
        return self.input == other.input and self.output == other.output

    @override
    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, Eblt):
            return False
        return self.input != other.input or self.output != other.output
