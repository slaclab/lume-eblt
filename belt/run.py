from __future__ import annotations

import logging
import os
import pathlib
import platform
import shlex
import shutil
import traceback
from time import monotonic
from typing import Any, ClassVar, Optional, Union, Sequence

import h5py
import psutil
from lume import tools as lume_tools
from lume.base import CommandWrapper
from pmd_beamphysics import ParticleGroup
from typing_extensions import override

from . import tools

from .output import RunInfo
from .fieldmap import read_fieldmap_rfdata, write_fieldmap_rfdata
from typing import List, Dict

from .input import BELTInput, assign_names_to_elements, DriftTube, Bend, RFCavity, Wakefield
from .output import BELTOutput
from .particles import BELTParticleData

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


# def _make_belt_input():
#    pass


class BELT(CommandWrapper):
    """ """

    COMMAND: ClassVar[str] = "xbelt"
    COMMAND_MPI: ClassVar[str] = "xbelt-par"
    MPI_RUN: ClassVar[str] = find_mpirun()
    WORKDIR: ClassVar[Optional[str]] = find_workdir()

    # Environmental variables to search for executables
    command_env: str = "BELT_BIN"
    command_mpi_env: str = "BELT_BIN"
    original_path: AnyPath

    _input: BELTInput
    output: Optional[BELTOutput]
    initial_particles: Optional[Union[ParticleGroup, BELTParticleData]] = None
    fieldmaps: List[Dict] = []

    def __init__(
            self,
            input: Optional[Union[BELTInput, str, pathlib.Path]] = None,
            *,
            workdir: Optional[Union[str, pathlib.Path]] = None,
            output: Optional[BELTOutput] = None,
            command: Optional[str] = None,
            command_mpi: Optional[str] = None,
            use_mpi: bool = False,
            mpi_run: str = "",
            use_temp_dir: bool = True,
            verbose: bool = tools.global_display_options.verbose >= 1,
            timeout: Optional[float] = None,
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

        if input:
            if not isinstance(input, BELTInput):
                # We have either a string or a filename for our main input.
                self.input_file_path, _ = os.path.split(input)
                input = BELTInput.from_file(input)
                assign_names_to_elements(input.lattice_lines)

        if workdir is None:
            workdir = pathlib.Path(".")

        self.original_path = workdir

        self._input = input
        self.output = output

        # MPI
        self.nproc = 1
        self.nnode = 1

    @property
    def input(self) -> BELTInput:
        """BELT input"""
        return self._input

    @input.setter
    def input(self, inp: Any) -> None:
        if not isinstance(inp, BELTInput):
            raise ValueError(
                f"The provided input is of type {type(inp).__name__} and not `BELTInput`. "
                f"Please consider creating a new BELT object instead with the "
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
    ) -> BELTOutput:
        """
        Execute BELT with the configured input settings.

        Parameters
        ----------
        load_particles : bool, default=False
            After execution, load all particle files.
        raise_on_error : bool, default=True
            If BELT fails to run, raise an error. Depending on the error,
            output information may still be accessible in the ``.output``
            attribute.

        Returns
        -------
        BELTOutput
            The output data.  This is also accessible as ``.output``.
        """
        if not self.configured:
            self.configure()

        if self.path is None:
            raise ValueError("Path (base_path) not yet set")

        self.finished = False

        runscript = self.get_run_script()

        start_time = monotonic()
        self.vprint(f"Running BELT in {self.path}")
        self.vprint(runscript)

        self.write_input()

        if self.timeout:
            self.vprint(
                f"Timeout of {self.timeout} is being used; output will be "
                f"displaye after BELT exits."
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
                log.append(f"BELT exited with an error: {ex}")
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
            self.output.lattice_lines = self._input.lattice_lines

        except Exception as ex:
            stack = traceback.format_exc()
            run_info.error = True
            run_info.error_reason = (
                f"Failed to load output file. {ex.__class__.__name__}: {ex}\n{stack}"
            )
            self.output = BELTOutput(run=run_info)
            if hasattr(ex, "add_note"):
                # Python 3.11+
                ex.add_note(
                    f"\nBELT output was:\n\n{execute_result['log']}\n(End of BELT output)"
                )
            if raise_on_error:
                raise

        self.output.run = run_info
        if run_info.error and raise_on_error:
            raise Exception(f"BELT failed to run: {run_info.error_reason}")

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

        # Todo:  update beam radius
        #  update beam radius given an external function
        # self.update_beam_radius(r, name)

        # write main input file
        filename = os.path.join(path, 'belt.in')
        self.input.to_file(filename=filename)

        # write run script
        if write_run_script:
            self.write_run_script(path)

    def write_initial_particles(self, path: Optional[AnyPath] = None) -> None:

        if self.initial_particles:
            Ek = self._input.parameters.Ek
            if isinstance(self.initial_particles, ParticleGroup):
                self.initial_particles = BELTParticleData.from_ParticleGroup(self.initial_particles, Ek)
            else:
                # If read from BELT output, shift the ref energy to the one defined in input file
                self.initial_particles.shift_ref_energy(Ek)

            self.initial_particles.write_BELT_input(path)
            # update header
            self._input.parameters.flagdist = 100
            self._input.parameters.np = self.initial_particles.np



        elif self._input.parameters.flagdist in [100, 200, 300]:
            src = os.path.join(self.input_file_path, 'pts.in')
            dest = os.path.join(path, 'pts.in')

            assert os.path.isfile(src), "Initial particles file not found"

            # Don't worry about overwriting in temporary directories
            if self._tempdir and os.path.exists(dest):
                os.remove(dest)

            if not os.path.exists(dest):
                shutil.copyfile(src, dest)
                # writers.write_input_particles_from_file(src, dest, self.header['Np'])
            else:
                self.vprint('pts.in already exits, will not overwrite.')

    def update_ref_energy(self, Ek: float) -> None:
        print("Updating Ek in the header and shifting the ref energy in particles.\n")
        print("Warning: The lattice parameters may need to be updated with the new ref energy")

        self._input.parameters.Ek = Ek
        self.initial_particles.shift_ref_energy(Ek)

    def update_beam_radius(self, r: float, name: str) -> None:
        print('Updating beam radius in the lattice element ', name, ' to be ', r)
        for lattice_element in self._input.lattice_lines:
            if (isinstance(lattice_element, DriftTube) or
                isinstance(lattice_element, Bend) or
                isinstance(lattice_element, RFCavity)) and lattice_element.name == name:
                lattice_element.V1 = r

    def load_wakefield(self, path: Optional[AnyPath] = None) -> None:
        # parse wakefield

        rec = []
        for lattice_element in self._input.lattice_lines:
            if isinstance(lattice_element, Wakefield):
                file_id = lattice_element.wake_function_file_id

                if file_id in rec:
                    continue

                self.fieldmaps.append(read_fieldmap_rfdata(self.input_file_path, file_id))
                rec.append(file_id)

    def write_wakefield(self, path: Optional[AnyPath] = None) -> None:

        self.load_wakefield(path=self.input_file_path)

        for fieldmap in self.fieldmaps:

            dest = os.path.join(path, fieldmap['info']['filename'])

            # Don't worry about overwriting in temporary directories
            if self._tempdir and os.path.exists(dest):
                os.remove(dest)

            if not os.path.exists(dest):
                write_fieldmap_rfdata(dest, fieldmap)

            else:
                self.vprint(fieldmap['info']['filename'] + ' already exits, will not overwrite.')

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
        self.input = BELTInput.from_archive(h5["input"])
        if "output" in h5:
            self.output = BELTOutput.from_archive(h5["output"])
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
    def from_archive(cls, arch: Union[AnyPath, h5py.Group]) -> BELT:
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
    def load_output(self) -> BELTOutput:
        return BELTOutput.from_directory(self.path)

    @override
    def plot(
            self,
            y: Union[str, Sequence[str]] = "kinetic_energy",
            x="distance",
            xlim=None,
            ylim=None,
            ylim2=None,
            yscale="linear",
            yscale2="linear",
            y2="rms_z",
            nice=True,
            include_layout=True,
            include_legend=True,
            return_figure=False,
            tex=False,
            **kwargs,
    ):
        """
        Plots output multiple keys.

        Parameters
        ----------
        y : list
            List of keys to be displayed on the Y axis
        x : str
            Key to be displayed as X axis
        xlim : list
            Limits for the X axis
        ylim : list
            Limits for the Y axis
        ylim2 : list
            Limits for the secondary Y axis
        yscale: str
            one of "linear", "log", "symlog", "logit", ... for the Y axis
        yscale2: str
            one of "linear", "log", "symlog", "logit", ... for the secondary Y axis
        y2 : list
            List of keys to be displayed on the secondary Y axis
        nice : bool
            Whether or not a nice SI prefix and scaling will be used to
            make the numbers reasonably sized. Default: True
        include_layout : bool
            Whether or not to include a layout plot at the bottom. Default: True
            Whether or not the plot should include the legend. Default: True
        return_figure : bool
            Whether or not to return the figure object for further manipulation.
            Default: True
        kwargs : dict
            Extra arguments can be passed to the specific plotting function.

        Returns
        -------
        fig : matplotlib.pyplot.figure.Figure
            The plot figure for further customizations or `None` if `return_figure` is set to False.
        """
        if self.output is None:
            raise RuntimeError(
                "Genesis 4 has not yet been run; there is no output to plot."
            )

        if not tools.is_jupyter():
            # If not in jupyter mode, return a figure by default.
            return_figure = True

        return self.output.plot(
            y=y,
            x=x,
            xlim=xlim,
            ylim=ylim,
            ylim2=ylim2,
            yscale=yscale,
            yscale2=yscale2,
            y2=y2,
            nice=nice,
            include_layout=include_layout,
            include_legend=include_legend,
            return_figure=return_figure,
            tex=tex,
            **kwargs,
        )

    def stat(self, key: str):
        if self.output is None:
            raise RuntimeError(
                "BELT has not yet been run; there is no output to get statistics from."
            )
        return self.output.stat(key=key)

    @property
    def input(self) -> BELTInput:
        """The Genesis 4 input, including namelists and lattice information."""
        return self._input

    @input.setter
    def input(self, inp: Any) -> None:
        if not isinstance(inp, BELTInput):
            raise ValueError(
                f"The provided input is of type {type(inp).__name__} and not `BELTInput`. "
                f"Please consider creating a new Genesis4 object instead with the "
                f"new parameters!"
            )
        self._input = inp

    @override
    @staticmethod
    def input_parser(path: AnyPath) -> BELTInput:
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
        return BELTInput.from_file(path)

    @override
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BELT):
            return False
        return self.input == other.input and self.output == other.output

    @override
    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, BELT):
            return False
        return self.input != other.input or self.output != other.output

    @override
    def fingerprint(self, digest_size=16):
        """
        Creates a cryptographic fingerprint from keyed data.
        Used JSON dumps to form strings, and the blake2b algorithm to hash.

        Parameters
        ----------
        keyed_data : dict
            dict with the keys to generate a fingerprint
        digest_size : int, optional
            Digest size for blake2b hash code, by default 16

        Returns
        -------
        str
            The hexadecimal digest
        """
        return self.input.fingerprint()
