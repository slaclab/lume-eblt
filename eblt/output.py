import os
import re
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Sequence, Type, Union, Annotated, Optional, Dict, List
import pydantic
import pydantic_core
from . import archive as _archive
import h5py
from .types import PydanticPmdUnit, NDArray
from .input import Bend, Chicane, DriftTube, RFCavity, WriteBeam, ChangeEnergy, Wakefield, Exit
from .plot import PlotMaybeLimits, PlotLimits, plot_stats_with_layout 
import matplotlib
from pmd_beamphysics.units import pmd_unit
from .particles import EBLTParticleData



class RunInfo(BaseModel):
    """
    run information.

    Attributes
    ----------
    error : bool
        True if an error occurred during the  run.
    error_reason : str or None
        Error explanation, if `error` is set.
    run_script : str
        The command-line arguments used to run
    output_log : str
        Genesis 4 output log
    start_time : float
        Start time of the process
    end_time : float
        End time of the process
    run_time : float
        Wall clock run time of the process
    """

    error: bool = pydantic.Field(
        default=False, description="`True` if an error occurred during the EBLT run"
    )
    error_reason: Optional[str] = pydantic.Field(
        default=None, description="Error explanation, if `error` is set."
    )
    run_script: str = pydantic.Field(
        default="", description="The command-line arguments used to run Genesis"
    )
    output_log: str = pydantic.Field(
        default="", repr=False, description="Genesis 4 output log"
    )
    start_time: float = pydantic.Field(
        default=0.0, repr=False, description="Start time of the process"
    )
    end_time: float = pydantic.Field(
        default=0.0, repr=False, description="End time of the process"
    )
    run_time: float = pydantic.Field(
        default=0.0, description="Wall clock run time of the process"
    )

    @property
    def success(self) -> bool:
        """`True` if the run was successful."""
        return not self.error

# Model for stats (formerly fort.2)
class StatsOutput(BaseModel):
    units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict, repr=False)
    distance: NDArray = Field(..., description="Distance along the beamline (m)")
    kinetic_energy: NDArray = Field(..., description="Kinetic energy (eV)")
    gamma: NDArray = Field(..., description="Relativistic gamma (1)")
    mean_z: NDArray = Field(..., description="Mean Z coordinate (m)")
    rms_z: NDArray = Field(..., description="RMS Z coordinate (m)")
    mean_delta_gamma: NDArray = Field(..., description="Mean Δγ")
    rms_delta_gamma: NDArray = Field(..., description="RMS Δγ")

    @classmethod
    def load_from_file(cls, filepath: str) -> "StatsOutput":
        data = np.loadtxt(filepath)
        data = np.atleast_2d(data)  # Ensure the data is always a 2D array
        units = dict()
        units["distance"] = pmd_unit("m")
        units["kinetic_energy"] = pmd_unit("eV")
        units["gamma"] = pmd_unit("1")
        units["mean_delta_gamma"] = pmd_unit("1")
        units["rms_delta_gamma"] = pmd_unit("1")
        units["mean_z"] = pmd_unit("m")
        units["rms_z"] = pmd_unit("m")
        return cls(
            distance=data[:, 0],
            kinetic_energy=data[:, 1],
            gamma=data[:, 2],
            mean_z=data[:, 3],
            rms_z=data[:, 4],
            mean_delta_gamma=data[:, 5],
            rms_delta_gamma=data[:, 6],
            units = units
        )


# Model for Current Profile Outputs (3 columns)
class CurrentProfileOutput(BaseModel):
    units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict, repr=False)
    bunch_length: NDArray = Field(..., description="Bunch length (m)")
    charge_per_cell: NDArray = Field(..., description="Charge per cell")
    current: NDArray = Field(..., description="Current (A)")

    @classmethod
    def load_from_file(cls, filepath: str) -> "CurrentProfileOutput":
        data = np.loadtxt(filepath)
        data = np.atleast_2d(data)  # Ensure the data is always a 2D array
        units = dict()
        units["bunch_length"] = pmd_unit("m")
        units["current"] = pmd_unit("A")
        return cls(
            bunch_length=data[:, 0], charge_per_cell=data[:, 1], current=data[:, 2], units = units
        )


# Model for Particle Distribution Outputs (4 columns)
#class ParticleDistributionOutput(BaseModel):
#    units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict, repr=False)
#    z: NDArray = Field(..., description="Z coordinate (m)")
#    delta_gamma: NDArray = Field(..., description="Δγ")
#    weight: NDArray = Field(..., description="Particle weight")
#    delta_e_over_e0: NDArray = Field(..., description="dE/E0")

#    @classmethod
#    def load_from_file(cls, filepath: str) -> "ParticleDistributionOutput":
#        data = np.loadtxt(filepath)
#        data = np.atleast_2d(data)  # Ensure the data is always a 2D array
#        units = dict()
#        units["z"] = pmd_unit("m")
#        units["delta_gamma"] = pmd_unit("1")
#        units["delta_e_over_e0"] = pmd_unit("1")
#        return cls(
#            z=data[:, 0],
#            delta_gamma=data[:, 1],
#            weight=data[:, 2],
#            delta_e_over_e0=data[:, 3],
#            units = units
            
#       )

#    @property
#    def gamma0(self):
#        return self.delta_gamma / self.delta_e_over_e0

#    @property
#    def gamma(self):
#        return self.gamma0 + self.delta_gamma


# Combined EBLTOutput object
class EBLTOutput(BaseModel):
    units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict, repr=False)
    stats: Optional[StatsOutput] = None
    current_profiles: Dict[int, CurrentProfileOutput] = Field(default_factory=dict)
    particle_distributions: Dict[int, EBLTParticleData] = Field(
        default_factory=dict
    )
    lattice_lines: List[Union[Bend, Chicane, DriftTube, RFCavity, WriteBeam, ChangeEnergy, Wakefield, Exit]] = None,
    run: RunInfo = Field(
        default_factory=RunInfo,
        description="Run-related information - output text and timing.",
    )

    @classmethod
    def from_directory(cls, directory: str) -> "EBLTOutput":
        output = {}
        units =  dict()
       
        # Handle the fort.2 file (stats)
        stats_file = os.path.join(directory, "fort.2")
        if os.path.exists(stats_file):
            output["stats"] = StatsOutput.load_from_file(stats_file)
            units.update(output["stats"].units)
   

        # Handle the fort.i and fort.i+1 files with auto-detection
        current_profiles = {}
        particle_distributions = {}
        for filename in os.listdir(directory):
            if filename == "fort.2":  # Skip fort.2 as it is handled separately
                continue
            match = re.match(r"fort\.(\d+)", filename)
            if match:
                i = int(match.group(1))
                filepath = os.path.join(directory, filename)
                data = np.loadtxt(filepath)
                data = np.atleast_2d(data)  # Ensure the data is always a 2D array
                if data.shape[1] == 3:  # CurrentProfileOutput has 3 columns
                    current_profiles[i] = CurrentProfileOutput.load_from_file(filepath)
                    units.update(current_profiles[i].units)
                    
                elif data.shape[1] == 4:  # ParticleDistributionOutput has 4 columns
                    particle_distributions[i] = (
                        EBLTParticleData.from_EBLT_outputfile(filepath)
                    )
                    #units.update(particle_distributions[i].units)
                   

        output["current_profiles"] = current_profiles
        output["particle_distributions"] = particle_distributions

        output["units"] = units

        return cls(**output)


    def plot(
        self,
        y: Union[str, Sequence[str]] = "kinetic_energy",
        x: str = "distance",
        xlim: Optional[PlotLimits] = None,
        ylim: Optional[PlotMaybeLimits] = None,
        ylim2: Optional[PlotMaybeLimits] = None,
        yscale: str = "linear",
        yscale2: str = "linear",
        y2: Union[str, Sequence[str]] = "rms_z",
        nice: bool = True,
        include_layout: bool = True,
        include_legend: bool = True,
        return_figure: bool = False,
        tex: bool = False,
        **kwargs,
    ) -> Optional[matplotlib.figure.Figure]:
        """
        Plots output multiple keys.

        Parameters
        ----------
        y : str or list of str
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

        # Expand keys
        if isinstance(y, str):
            y = y.split()
        if isinstance(y2, str):
            y2 = y2.split()

        y = list(y)
        y2 = list(y2)

        return plot_stats_with_layout(
            self,
            ykeys=y,
            ykeys2=y2,
            xkey=x,
            xlim=xlim,
            ylim=ylim,
            ylim2=ylim2,
            yscale=yscale,
            yscale2=yscale2,
            nice=nice,
            tex=tex,
            include_layout=include_layout,
            include_legend=include_legend,
            return_figure=return_figure,
            **kwargs,
        )

    def plot_distribution(self, file_id: int, xkey: str, ykey:str, bins: int = 50)->None:
        self.particle_distributions[file_id].plot(xkey = 't', ykey = 'energy', bins = 100)

    def archive(self, h5: h5py.Group) -> None:
        """
        Dump outputs into the given HDF5 group.

        Parameters
        ----------
        h5 : h5py.Group
            The HDF5 file in which to write the information.
        """
        _archive.store_in_hdf5_file(h5, self)

    @classmethod
    def from_archive(cls, h5: h5py.Group) -> "EBLTOutput":
        """
        Loads output from the given HDF5 group.

        Parameters
        ----------
        h5 : str or h5py.File
            The key to use when restoring the data.
        """
        loaded = _archive.restore_from_hdf5_file(h5)
        if not isinstance(loaded, EBLTOutput):
            raise ValueError(
                f"Loaded {loaded.__class__.__name__} instead of a "
                f"Genesis4Output instance.  Was the HDF group correct?"
            )
        return loaded



# Example Usage
if __name__ == "__main__":
    try:
        # Load all relevant data from a directory containing the fort.* files
        eblt_output = EBLTOutput.from_directory("path/to/your/directory")

        # Access the loaded data
        print(eblt_output.stats)
        print(eblt_output.current_profiles)
        print(eblt_output.particle_distributions)
    except ValidationError as e:
        print("Validation Error:", e)
    except ValueError as e:
        print("Value Error:", e)