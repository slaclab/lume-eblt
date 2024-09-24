import os
import re
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Sequence, Type, Union, Annotated, Optional, Dict
import pydantic
import pydantic_core
from . import archive as _archive
import h5py

# Custom Pydantic class to handle numpy ndarray
class _PydanticNDArray:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: Type[Any],
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.core_schema.CoreSchema:
        def serialize(
            obj: np.ndarray, info: pydantic_core.core_schema.SerializationInfo
        ):
            if not isinstance(obj, np.ndarray):
                raise ValueError(
                    f"Only supports numpy ndarray. Got {type(obj).__name__}: {obj}"
                )
            return obj.tolist()

        return pydantic_core.core_schema.with_info_plain_validator_function(
            cls._pydantic_validate,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                serialize, when_used="json-unless-none", info_arg=True
            ),
        )

    @classmethod
    def _pydantic_validate(
        cls,
        value: Union[Any, np.ndarray, Sequence, dict],
        info: pydantic_core.core_schema.ValidationInfo,
    ) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, Sequence):
            return np.asarray(value)
        raise ValueError(f"No conversion from {value!r} to numpy ndarray")


# Annotate numpy arrays to be handled by the custom class
NDArray = Annotated[np.ndarray, _PydanticNDArray]


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
    run: RunInfo = Field(
        default_factory=RunInfo,
        description="Run-related information - output text and timing.",
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