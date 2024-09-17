from pydantic import BaseModel
import h5py
from pydantic import BaseModel, Field
from .types import AnyPath, NDArray
from pmd_beamphysics import ParticleGroup
import numpy as np

class EBLTParticleData(BaseModel):
    """

    """
    z: NDArray = Field(..., description="Z coordinate (m)")
    delta_gamma: NDArray = Field(..., description="Δγ")
    weight: NDArray = Field(..., description="Particle weight")
    delta_e_over_e0: NDArray = Field(..., description="dE/E0")


    @classmethod
    def from_ParticleGroup(cls, pg: ParticleGroup) -> "EBLTParticleData":
        return cls(
            z = pg.z,
            delta_gamma = pg.gamma - pg.
        )

    @classmethod
    def from_EBLT_outputfile(cls, filepath: AnyPath) -> "EBLTParticleData":
        data = np.loadtxt(filepath)
        data = np.atleast_2d(data)  # Ensure the data is always a 2D array
        return cls(
            z=data[:, 0],
            delta_gamma=data[:, 1],
            weight=data[:, 2],
            delta_e_over_e0=data[:, 3],
        )

    @classmethod
    def from_ImpactT_outputfile(cls, path: AnyPath) -> "EBLTParticleData":
        pass

    @classmethod
    def write_EBLT_input(cls, path: AnyPath, verbose: bool = True) -> None:
        pass







#class ImactTSliceData(BaseModel):
#    pass

