import os
import re
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Sequence, Type, Union, Annotated, Optional, Dict
import pydantic
import pydantic_core
from .types import NDArray

# Model for stats (formerly fort.2)
class StatsOutput(BaseModel):
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
        return cls(
            distance=data[:, 0],
            kinetic_energy=data[:, 1],
            gamma=data[:, 2],
            mean_z=data[:, 3],
            rms_z=data[:, 4],
            mean_delta_gamma=data[:, 5],
            rms_delta_gamma=data[:, 6],
        )


# Model for Current Profile Outputs (3 columns)
class CurrentProfileOutput(BaseModel):
    bunch_length: NDArray = Field(..., description="Bunch length (m)")
    charge_per_cell: NDArray = Field(..., description="Charge per cell")
    current: NDArray = Field(..., description="Current (A)")

    @classmethod
    def load_from_file(cls, filepath: str) -> "CurrentProfileOutput":
        data = np.loadtxt(filepath)
        data = np.atleast_2d(data)  # Ensure the data is always a 2D array
        return cls(
            bunch_length=data[:, 0], charge_per_cell=data[:, 1], current=data[:, 2]
        )


# Model for Particle Distribution Outputs (4 columns)
class ParticleDistributionOutput(BaseModel):
    z: NDArray = Field(..., description="Z coordinate (m)")
    delta_gamma: NDArray = Field(..., description="Δγ")
    weight: NDArray = Field(..., description="Particle weight")
    delta_e_over_e0: NDArray = Field(..., description="dE/E0")

    @classmethod
    def load_from_file(cls, filepath: str) -> "ParticleDistributionOutput":
        data = np.loadtxt(filepath)
        data = np.atleast_2d(data)  # Ensure the data is always a 2D array
        return cls(
            z=data[:, 0],
            delta_gamma=data[:, 1],
            weight=data[:, 2],
            delta_e_over_e0=data[:, 3],
        )

    @property
    def gamma0(self):
        return self.delta_gamma / self.delta_e_over_e0

    @property
    def gamma(self):
        return self.gamma0 + self.delta_gamma


# Combined EBLTOutput object
class EBLTOutput(BaseModel):
    stats: Optional[StatsOutput] = None
    current_profiles: Dict[int, CurrentProfileOutput] = Field(default_factory=dict)
    particle_distributions: Dict[int, ParticleDistributionOutput] = Field(
        default_factory=dict
    )

    @classmethod
    def from_directory(cls, directory: str) -> "EBLTOutput":
        output = {}

        # Handle the fort.2 file (stats)
        stats_file = os.path.join(directory, "fort.2")
        if os.path.exists(stats_file):
            output["stats"] = StatsOutput.load_from_file(stats_file)

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
                elif data.shape[1] == 4:  # ParticleDistributionOutput has 4 columns
                    particle_distributions[i] = (
                        ParticleDistributionOutput.load_from_file(filepath)
                    )

        output["current_profiles"] = current_profiles
        output["particle_distributions"] = particle_distributions

        return cls(**output)


class RunInfo(BaseModel):
    """
    run information.

    Attributes
    ----------
    error : bool
        True if an error occurred during the run.
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
        default=False, description="`True` if an error occurred during the Genesis run"
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
