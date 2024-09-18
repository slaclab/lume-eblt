from pydantic import BaseModel
import h5py
from pydantic import BaseModel, Field
from .types import AnyPath, NDArray
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.interfaces.impact import  impact_particles_to_particle_data
import numpy as np
from pmd_beamphysics.units import mec2
import os


def parse_impact_particles(filePath,
                           names=('x', 'GBx', 'y', 'GBy', 'z', 'GBz'),
                           skiprows=0):
    """
    Parse Impact-T input and output particle data.
    Typical filenames: 'partcl.data', 'fort.40', 'fort.50'.

    Note that partcl.data has the number of particles in the first line, so skiprows=1 should be used.

    Returns a structured numpy array

    Impact-T input/output particles distribions are ASCII files with columns:
    x (m)
    GBy = gamma*beta_x (dimensionless)
    y (m)
    GBy = gamma*beta_y (dimensionless)
    z (m)
    GBz = gamma*beta_z (dimensionless)

    """

    dtype = {'names': names,
             'formats': 6 * [float]}
    pdat = np.loadtxt(filePath, skiprows=skiprows, dtype=dtype,
                      ndmin=1)  # to make sure that 1 particle is parsed the same as many.

    return pdat

class EBLTParticleData(BaseModel):
    """

    """
    z: NDArray = Field(..., description="Z coordinate (m)")
    delta_gamma: NDArray = Field(..., description="Δγ")
    weight: NDArray = Field(..., description="Particle weight")
    delta_e_over_e0: NDArray = Field(..., description="dE/E0")
    beam_radius: float = None


    @classmethod
    def from_ParticleGroup(cls, pg: ParticleGroup) -> "EBLTParticleData":
        return cls(
            z = pg.z,
            delta_gamma = pg.gamma - pg.avg('gamma'),
            delta_e_over_e0 = (pg['energy'] - pg['mean_energy'])/pg['mean_energy'],
            weight = pg.weight,
            beam_radius = np.sqrt(pg['sigma_x']**2 + pg['sigma_y']**2)
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
    def from_ImpactT_outputfile(cls, path: AnyPath,
                                mc2: float = mec2, species: str = 'electron') -> "EBLTParticleData":
        tout = parse_impact_particles(path)
        data = impact_particles_to_particle_data(tout, mc2, species)
        pg = ParticleGroup(data=data)
        return cls.from_ParticleGroup(pg)


    def to_particlegroup_data(self) ->ParticleGroup:
        z = self.z
        gamma = self.gamma
        weight = self.weight
        n = len(z)
        pz = np.sqrt(gamma**2 - 1) * mec2
        particlegroup_data = dict(  t=np.zeros(n),
                                    x=np.zeros(n),
                                    px=np.zeros(n),
                                    y=np.zeros(n),
                                    py=np.zeros(n),
                                    z=self.z,
                                    pz=pz,
                                    weight=weight,
                                    status=np.ones(n),
                                    species="electron",
                                )
        return ParticleGroup(data= particlegroup_data)


    def write_EBLT_input(self, path: AnyPath, verbose: bool = True) -> None:
        data = np.vstack(self.z, self.delta_gamma, self.weight, self.delta_e_over_e0).T
        file_path = os.path.join(path, 'pts.in')
        np.savetxt(file_path, data)


    @property
    def gamma0(self):
        return self.delta_gamma / self.delta_e_over_e0

    @property
    def gamma(self):
        return self.gamma0 + self.delta_gamma







#class ImactTSliceData(BaseModel):
#    pass

