from pydantic import BaseModel
import h5py
from pydantic import BaseModel, Field
from .types import AnyPath, NDArray
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.interfaces.impact import  impact_particles_to_particle_data
import numpy as np
from pmd_beamphysics.units import mec2, c_light
import os
from typing import Optional



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
    Ek: Optional[float] = Field(None, description="Electron reference energy")
    beam_radius: Optional[float] = None
   

    @classmethod
    def from_ParticleGroup(cls, pg: ParticleGroup, Ek: float) -> "EBLTParticleData":
        return cls(
            z = pg.z,
            delta_gamma = pg.gamma - Ek/mec2,
            delta_e_over_e0 = (pg['energy'] - Ek)/Ek,
            weight = pg.weight,
            Ek = Ek,
            beam_radius = np.sqrt(pg['sigma_x']**2 + pg['sigma_y']**2)
        )

    @classmethod
    def from_EBLT_outputfile(cls, filepath: AnyPath, Ek: float = None) -> "EBLTParticleData":
        data = np.loadtxt(filepath)
        data = np.atleast_2d(data)  # Ensure the data is always a 2D array

        # Update delta_gamma and delta_e_over_e0 given the new Ek

       


        output = cls(z=data[:, 0],
            delta_gamma=data[:, 1],
            weight=data[:, 2],
            delta_e_over_e0=data[:, 3])
        
        if Ek:
            output.shift_ref_energy(Ek)

        return output


    def shift_ref_energy(self,  Ek: float) ->None:
        print('Shifting delta_e_over_e0 and delta_gamma given Ek')
        self.delta_gamma = self.gamma - Ek/mec2
        self.delta_e_over_e0 = self.delta_gamma /(Ek/mec2)
        self.Ek = Ek

    @classmethod
    def from_ImpactT_outputfile(cls, path: AnyPath, Ek: float,
                                mc2: float = mec2, species: str = 'electron') -> "EBLTParticleData":
        tout = parse_impact_particles(path)
        data = impact_particles_to_particle_data(tout, mc2, species)
        pg = ParticleGroup(data=data)
        return cls.from_ParticleGroup(pg, Ek)


    def to_particlegroup(self) ->ParticleGroup:
        z = self.z
        gamma = self.gamma
        weight = self.weight
        n = len(z)
        pz = np.sqrt(gamma**2 - 1) * mec2
        particlegroup_data = dict(  t=self.z/c_light,
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

    def plot(self, xkey: str, ykey:str, bins: int = 50) -> None:
        self.to_particlegroup().plot(xkey, ykey, bins = bins)


    def write_EBLT_input(self, path: AnyPath, verbose: bool = True) -> None:
        data = np.vstack((self.z, self.delta_gamma, self.weight, self.delta_e_over_e0)).T
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

