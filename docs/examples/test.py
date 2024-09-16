from eblt.input import EBLTInput, assign_names_to_elements
from eblt.output import EBLTOutput
from eblt.plot import plot_lattice_lines

import numpy as np

from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import mec2
import rich
import matplotlib.pyplot as plt

input = EBLTInput.from_file("example1/eblt.in")
assign_names_to_elements(input.lattice_lines)