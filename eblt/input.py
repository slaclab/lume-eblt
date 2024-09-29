import re
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Union
from . import archive as _archive
import h5py
import shlex
import pathlib
from lume import tools as lume_tools
from .tools import class_key_data
from typing_extensions import override
from hashlib import blake2b
from .tools import update_hash

# Define the base classes for coefficients and lattice elements
class Parameters(BaseModel):
    np: int = Field(..., description="Number of macroparticles")
    nz: int = Field(..., description="Number of longitudinal grid poisnts")
    zmin: float = Field(..., description="Minimum z-coordinate (m)")
    zmax: float = Field(..., description="Maximum z-coordinate (m)")
    flagfwd: int = Field(..., description="Flag for forward tracking")
    flagdist: int = Field(..., description="Switch for different type of initial distributions")
    Iavg: float = Field(..., description="Average current (A)")
    Ek: float = Field(..., description="Kinetic energy (eV)")
    mass: float = Field(..., description="Mass (MeV/c^2)")
    charge: float = Field(..., description="Charge (C)")
    freq: float = Field(..., description="Frequency (Hz)")


class CoefficientsLine(BaseModel):
    coefficients: List[float] = Field(..., description="List of coefficients")


class Icoefficients(CoefficientsLine):
    pass


class PhaseSpaceCoefficients(CoefficientsLine):
    pass


# Utility class for initial parsing of lattice elements
class LatticeElement(BaseModel):
    length: float
    Bnseg: Optional[int]
    Bmpstp: Optional[int]
    Btype: int
    V: List[float]
    name: Optional[str]

    @model_validator(mode="before")
    def cast_to_int(cls, values):
        if "Bnseg" in values and values["Bnseg"] is not None:
            values["Bnseg"] = int(values["Bnseg"])
        if "Bmpstp" in values and values["Bmpstp"] is not None:
            values["Bmpstp"] = int(values["Bmpstp"])
        return values


# Define specific lattice element classes with named parameters for V values
class DriftTube(BaseModel):
    length: float = Field(..., description="Length of the drift tube (m)")
    beam_radius: float = Field(
        ..., description="Beam radius (m) used in longitudinal space-charge"
    )
    name: Optional[str] = Field(
        None, description="Optional name for the drift tube element"
    )

    @classmethod
    def from_lattice_element(cls, lattice_element: LatticeElement):
        return cls(
            length=lattice_element.length,
            beam_radius=lattice_element.V[0],
            name=lattice_element.name,
        )

    def to_lattice_element(self) -> LatticeElement:
        return LatticeElement(
            length=self.length,
            Bnseg=1,
            Bmpstp=1,
            Btype=0,
            V=[self.beam_radius],
            name=self.name,
        )


class Bend(BaseModel):
    length: float = Field(..., description="Length of the bend element (m)")
    beam_radius: float = Field(
        ..., description="Beam radius (m) used in longitudinal space-charge"
    )
    angle: float = Field(..., description="Bending angle (radians)")
    CSR_switch: Optional[float] = Field(
        None,
        description="Switch for CSR (1.01 -> IGF CSR including A and B, 2.01 -> IGF steady-state CSR, 0.0 -> no CSR)",
    )
    SC_switch: Optional[float] = Field(
        None, description="Switch for space charge (1 on, otherwise off)"
    )
    name: Optional[str] = Field(None, description="Optional name for the bend element")

    @classmethod
    def from_lattice_element(cls, lattice_element: LatticeElement):
        return cls(
            length=lattice_element.length,
            beam_radius=lattice_element.V[0],
            angle=lattice_element.V[4],
            CSR_switch=lattice_element.V[6] if len(lattice_element.V) > 6 else None,
            SC_switch=lattice_element.V[7] if len(lattice_element.V) > 7 else None,
            name=lattice_element.name,
        )

    def to_lattice_element(self) -> LatticeElement:
        V = [self.beam_radius]

        # Ensure proper index for angle, CSR, and SC switches
        while len(V) < 5:
            V.append(0)  # Fill with placeholders up to V5
        V[4] = self.angle

        if self.CSR_switch is not None:
            V.append(0)  # Placeholder for V6
            V.append(self.CSR_switch)
        if self.SC_switch is not None:
            if len(V) < 7:
                V.append(0)  # Ensure correct index for SC_switch
            V.append(self.SC_switch)

        return LatticeElement(
            length=self.length, Bnseg=1, Bmpstp=1, Btype=4, V=V, name=self.name
        )


class Chicane(BaseModel):
    length: float = Field(..., description="Length of the chicane element (m)")
    beam_radius: float = Field(
        ..., description="Beam radius (m) used in longitudinal space-charge"
    )
    drift_length: float = Field(
        ..., description="Drift length between first and second bends (m)"
    )
    angle: float = Field(..., description="Bending angle (radians)")
    R56: Optional[float] = Field(None, description="Transfer matrix element R56 (m)")
    T566: Optional[float] = Field(None, description="Transfer matrix element T566 (m)")
    U5666: Optional[float] = Field(
        None, description="Transfer matrix element U5666 (m)"
    )
    CSR_switch: Optional[float] = Field(
        None,
        description="Switch for CSR (1.01 -> IGF CSR including A and B, 2.01 -> IGF steady-state CSR, 0.0 -> no CSR)",
    )
    SC_switch: Optional[float] = Field(
        None, description="Switch for space charge (1 on, otherwise off)"
    )
    name: Optional[str] = Field(
        None, description="Optional name for the chicane element"
    )

    @classmethod
    def from_lattice_element(cls, lattice_element: LatticeElement):
        return cls(
            length=lattice_element.length,
            beam_radius=lattice_element.V[0],
            drift_length=lattice_element.V[1] if len(lattice_element.V) > 2 else None,     #From Manual it seems that R56t and drift length share V
            R56=lattice_element.V[1] if len(lattice_element.V) > 2 else None,
            T566=lattice_element.V[2] if len(lattice_element.V) > 2 else None,
            U5666=lattice_element.V[3] if len(lattice_element.V) > 3 else None,
            angle=lattice_element.V[4],
            CSR_switch=lattice_element.V[5] if len(lattice_element.V) > 5 else None,
            SC_switch=lattice_element.V[6] if len(lattice_element.V) > 6 else None,
            name=lattice_element.name,
        )

    def to_lattice_element(self) -> LatticeElement:
        V = [
            self.beam_radius,
            self.drift_length,
            #self.R56 or 0,
            self.T566 or 0,
            self.U5666 or 0,
            self.angle,
        ]

        # Append CSR and SC switches
        if self.CSR_switch is not None:
            V.append(self.CSR_switch)
        if self.SC_switch is not None:
            V.append(self.SC_switch)

        return LatticeElement(
            length=self.length,
            Bnseg=1,
            Bmpstp=1,  # Indicate chicane
            Btype=4,
            V=V,
            name=self.name,
        )


class RFCavity(BaseModel):
    length: float = Field(..., description="Length of the RF cavity element (m)")
    beam_radius: float = Field(
        ..., description="Beam radius (m) used in longitudinal space-charge"
    )
    gradient: float = Field(..., description="Gradient of the RF cavity (MV/m)")
    frequency: float = Field(..., description="Frequency of the RF cavity (Hz)")
    phase_deg: float = Field(..., description="Phase of the RF cavity in degrees")
    name: Optional[str] = Field(
        None, description="Optional name for the RF cavity element"
    )

    @classmethod
    def from_lattice_element(cls, lattice_element: LatticeElement):
        return cls(
            length=lattice_element.length,
            beam_radius=lattice_element.V[0],
            gradient=lattice_element.V[1],
            frequency=lattice_element.V[2],
            phase_deg=lattice_element.V[3],
            name=lattice_element.name,
        )

    def to_lattice_element(self) -> LatticeElement:
        return LatticeElement(
            length=self.length,
            Bnseg=1,
            Bmpstp=1,
            Btype=103,
            V=[self.beam_radius, self.gradient, self.frequency, self.phase_deg],
            name=self.name,
        )


class WriteBeam(BaseModel):
    iwrite: int = Field(
        None, description="integer for fort.i, fort.i+1 to write files to"
    )
    sample: int = Field(1, description="Stride for sampling particles. ")
    name: Optional[str] = Field(
        None, description="Optional name for the write beam element"
    )

    @classmethod
    def from_lattice_element(cls, lattice_element: LatticeElement):
        return cls(
            name=lattice_element.name,
            iwrite=lattice_element.Bmpstp,
            sample=lattice_element.V[0],
        )

    def to_lattice_element(self) -> LatticeElement:
        return LatticeElement(
            length=0,  # Length should always be zero
            Bnseg=1,
            Bmpstp=self.iwrite,
            Btype=-2,
            V=[self.sample],
            name=self.name,
        )


class ChangeEnergy(BaseModel):
    energy_increment: float = Field(..., description="Energy increment (eV)")
    name: Optional[str] = Field(
        None, description="Optional name for the change energy element"
    )

    @classmethod
    def from_lattice_element(cls, lattice_element: LatticeElement):
        return cls(energy_increment=lattice_element.V[0], name=lattice_element.name)

    def to_lattice_element(self) -> LatticeElement:
        return LatticeElement(
            length=0,  # Length should always be zero
            Bnseg=1,
            Bmpstp=1,
            Btype=-39,
            V=[self.energy_increment],
            name=self.name,
        )


class Wakefield(BaseModel):
    length: float = Field(..., description="Length of the wakefield element (m)")
    multiplier: float = Field(..., description="Wakefield multiplier")
    wake_function_file_id: float = Field(..., description="Wake function file ID")
    switch: float = Field(..., description="Switch for wakefield (1 on, otherwise off)")
    name: Optional[str] = Field(
        None, description="Optional name for the wakefield element"
    )

    @classmethod
    def from_lattice_element(cls, lattice_element: LatticeElement):
        return cls(
            length=lattice_element.length,
            multiplier=lattice_element.V[0],
            wake_function_file_id=lattice_element.V[1],
            switch=lattice_element.V[2],
            name=lattice_element.name,
        )

    def to_lattice_element(self) -> LatticeElement:
        return LatticeElement(
            length=self.length,
            Bnseg=1,
            Bmpstp=1,
            Btype=-41,
            V=[self.multiplier, self.wake_function_file_id, self.switch],
            name=self.name,
        )


class Exit(BaseModel):
    name: Optional[str] = Field(None, description="Optional name for the exit element")

    @classmethod
    def from_lattice_element(cls, lattice_element: LatticeElement):
        return cls(name=lattice_element.name)

    def to_lattice_element(self) -> LatticeElement:
        return LatticeElement(
            length=0,  # Length should always be zero
            Bnseg=1,
            Bmpstp=1,
            Btype=-99,
            V=[],
            name=self.name,
        )


# Define the main class for handling the input file
class EBLTInput(BaseModel):
    parameters: Parameters = Field(..., description="Simulation parameters")
    phase_space_coefficients: PhaseSpaceCoefficients = Field(
        ..., description="Phase space coefficients"
    )
    current_coefficients: Icoefficients = Field(
        ..., description="Current profile coefficients"
    )
    lattice_lines: List[
        Union[
            DriftTube, Bend, Chicane, RFCavity, WriteBeam, ChangeEnergy, Wakefield, Exit
        ]
    ] = Field(..., description="List of lattice elements")

    @staticmethod
    def preprocess_line(line: str) -> str:
        line = line.replace("d", "e")  # Replace 'd' with 'e' in scientific notation
        return line

    @staticmethod
    def format_value(value: float) -> str:
        """Formats the value to full precision, simplifying if possible."""
        if value.is_integer():
            return str(int(value))
        else:
            return f"{value:.15g}"  # Uses general format with 15 significant digits

    @classmethod
    def parse_lattice_element(
        cls, lattice_values: List[float], name: Optional[str]
    ) -> Union[
        Bend, Chicane, DriftTube, RFCavity, WriteBeam, ChangeEnergy, Wakefield, Exit
    ]:
        lattice_element = LatticeElement(
            length=lattice_values[0],
            Bnseg=lattice_values[1] if len(lattice_values) > 1 else None,
            Bmpstp=lattice_values[2] if len(lattice_values) > 2 else None,
            Btype=int(lattice_values[3]),
            V=lattice_values[4:],
            name=name,
        )

        if lattice_element.Btype == 0:
            return DriftTube.from_lattice_element(lattice_element)
        elif lattice_element.Btype == 4:
            if lattice_element.Bmpstp > 0:
                return Chicane.from_lattice_element(lattice_element)
            else:
                return Bend.from_lattice_element(lattice_element)
        elif lattice_element.Btype == 103:
            return RFCavity.from_lattice_element(lattice_element)
        elif lattice_element.Btype == -2:
            return WriteBeam.from_lattice_element(lattice_element)
        elif lattice_element.Btype == -39:
            return ChangeEnergy.from_lattice_element(lattice_element)
        elif lattice_element.Btype == -41:
            return Wakefield.from_lattice_element(lattice_element)
        elif lattice_element.Btype == -99:
            return Exit.from_lattice_element(lattice_element)
        else:
            raise ValueError(f"Unknown Btype: {lattice_element.Btype}")

    @classmethod
    def parse_from_lines(cls, lines: List[str]) -> "EBLTInput":
        parameter_lines = []
        lattice_lines = []

        for i, line in enumerate(lines):
            if not line.strip() or line.startswith("!"):
                continue

            line = cls.preprocess_line(line)
            main_part, _, comment = line.partition("/")
            comment = comment.strip()

            if i < 4:  # First four lines are parameter lines
                parameter_lines.append(
                    [
                        float(num) if "." in num or "e" in num else int(num)
                        for num in main_part.split()
                    ]
                )
            else:  # The remaining lines are lattice lines
                lattice_parts = main_part.split()
                lattice_values = [
                    float(num) if "." in num or "e" in num else int(num)
                    for num in lattice_parts
                ]
                name = None
                if "name:" in comment:
                    name_match = re.search(r"name:\s*(\w+)", comment)
                    if name_match:
                        name = name_match.group(1)

                lattice_element = cls.parse_lattice_element(lattice_values, name)
                lattice_lines.append(lattice_element)

        parameters = Parameters(
            np=parameter_lines[0][0],
            nz=parameter_lines[0][1],
            zmin=parameter_lines[0][2],
            zmax=parameter_lines[0][3],
            flagfwd=parameter_lines[0][4],
            flagdist=parameter_lines[0][5],
            Iavg=parameter_lines[3][0],
            Ek=parameter_lines[3][1],
            mass=parameter_lines[3][2],
            charge=parameter_lines[3][3],
            freq=parameter_lines[3][4],
        )

        return cls(
            parameters=parameters,
            current_coefficients=Icoefficients(coefficients=parameter_lines[1]),
            phase_space_coefficients=PhaseSpaceCoefficients(
                coefficients=parameter_lines[2]
            ),
            lattice_lines=lattice_lines,
        )

    def to_lines(self) -> List[str]:
        lines = []

        # Parameter lines with labels
        params = self.parameters
        lines.append("! np nz zmin zmax flagfwd flagdist")
        lines.append(
            f"{params.np} {params.nz} {self.format_value(params.zmin)} {self.format_value(params.zmax)} {params.flagfwd} {params.flagdist} /"
        )

        lines.append("! a0 a1 a2 a3 a4 a5 a6 a7 a8 a9")
        lines.append(
            " ".join(map(self.format_value, self.current_coefficients.coefficients))
            + " /"
        )

        lines.append("! b0 b1 b2 b3 b4 b5 b6 b7 b8 b9")
        lines.append(
            " ".join(map(self.format_value, self.phase_space_coefficients.coefficients))
            + " /"
        )

        lines.append("! Iavg Ek mass charge freq")
        lines.append(
            f"{self.format_value(params.Iavg)} {self.format_value(params.Ek)} {self.format_value(params.mass)} {self.format_value(params.charge)} {self.format_value(params.freq)} /"
        )

        # Lattice lines with customized labels
        for element in self.lattice_lines:
            lattice_element = element.to_lattice_element()

            # Create the label line based on the element type and its attributes
            if isinstance(element, Bend):
                label_line = (
                    "! length Bnseg Bmpstp Bend beam_radius angle CSR_switch SC_switch"
                )
            elif isinstance(element, Chicane):
                label_line = "! length Bnseg Bmpstp Chicane beam_radius drift_length R56 T566 U5666 angle CSR_switch SC_switch"
            elif isinstance(element, DriftTube):
                label_line = "! length Bnseg Bmpstp Drift"
            elif isinstance(element, RFCavity):
                label_line = "! length Bnseg Bmpstp RFCavity beam_radius gradient frequency phase_deg"
            elif isinstance(element, WriteBeam):
                label_line = "! length Bnseg Bmpstp WriteBeam"
            elif isinstance(element, ChangeEnergy):
                label_line = "! length Bnseg Bmpstp ChangeEnergy energy_increment"
            elif isinstance(element, Wakefield):
                label_line = "! length Bnseg Bmpstp Wakefield multiplier wake_function_file_id switch"
            elif isinstance(element, Exit):
                label_line = "! length Bnseg Bmpstp Exit"
            else:
                label_line = "! length Bnseg Bmpstp Unknown"

            if lattice_element.name:
                label_line += " name"

            lattice_str = f"{self.format_value(lattice_element.length)} {lattice_element.Bnseg} {lattice_element.Bmpstp} {lattice_element.Btype}"
            lattice_str += " " + " ".join(map(self.format_value, lattice_element.V))
            lattice_str += " /"

            if lattice_element.name:
                lattice_str += f" !name: {lattice_element.name}"

            lines.append(label_line)
            lines.append(lattice_str.strip())

        return lines

    @classmethod
    def from_file(cls, filename: str) -> "EBLTInput":
        with open(filename, "r") as file:
            lines = file.readlines()
        return cls.parse_from_lines(lines)

    def to_file(self, filename: str):
        with open(filename, "w") as file:
            lines = self.to_lines()
            file.write("\n".join(lines) + "\n")

    def archive(self, h5: h5py.Group) -> None:
        """
        Dump input data into the given HDF5 group.

        Parameters
        ----------
        h5 : h5py.Group
            The HDF5 file in which to write the information.
        """
        _archive.store_in_hdf5_file(h5, self)

    @classmethod
    def from_archive(cls, h5: h5py.Group) -> "EBLTInput":
        """
        Loads input from archived h5 file.

        Parameters
        ----------
        h5 : str or h5py.File
            The filename or handle on h5py.File from which to load data.
        """
        loaded = _archive.restore_from_hdf5_file(h5)
        if not isinstance(loaded, EBLTInput):
            raise ValueError(
                f"Loaded {loaded.__class__.__name__} instead of a "
                f"EBLTInput instance.  Was the HDF group correct?"
            )
        return loaded


    @property
    def arguments(self) -> List[str]:
        """
        Get all of the command-line arguments for running Genesis 4.

        Returns
        -------
        list of str
            Individual arguments to pass to Genesis 4.
        """
        optional_args = []
       
        return [*optional_args]

    def write_run_script(
        self,
        path: pathlib.Path,
        command_prefix: str = "xeblt",
    ) -> None:
        with open(path, mode="wt") as fp:
            print(shlex.join(shlex.split(command_prefix) + self.arguments), file=fp)
        lume_tools.make_executable(str(path))
    
    @override
    def fingerprint(self, digest_size=16):
        h = blake2b(digest_size=16)
        for key in ['parameters', 'phase_space_coefficients', 'current_coefficients']:
            keyed_data = class_key_data(getattr(self, key))
            update_hash(keyed_data, h)
        for element in self.lattice_lines:
            keyed_data = class_key_data(element)
            update_hash(keyed_data, h)
        return h.hexdigest()


def assign_names_to_elements(
    lattice_lines: List[
        Union[
            Bend, Chicane, DriftTube, RFCavity, WriteBeam, ChangeEnergy, Wakefield, Exit
        ]
    ],
):
    name_counters = {
        Bend: 1,
        Chicane: 1,
        DriftTube: 1,
        RFCavity: 1,
        WriteBeam: 1,
        ChangeEnergy: 1,
        Wakefield: 1,
        Exit: 1,
    }

    existing_names = set()

    for element in lattice_lines:
        if element.name is None:
            element_type = type(element)
            base_name = element_type.__name__.lower()
            while True:
                potential_name = f"{base_name}{name_counters[element_type]}"
                name_counters[element_type] += 1
                if potential_name not in existing_names:
                    element.name = potential_name
                    existing_names.add(potential_name)
                    break
        else:
            existing_names.add(element.name)


def test_eblt_interface():
    input_lines = [
        "100 200 0.0d0 1.0d0 1 2 /",
        "1.0 0.1 0.01 0.001 /",
        "0.0 0.0 0.0 0.0 /",
        "0.13 9.27065e+07 511005 -1.0 1.3e+09 /",
        "1.0 10 5 4 0.5 0.5 0.1 0.0 1.01 1 / !name: test",
    ]

    eblt_input = EBLTInput.parse_from_lines(input_lines)
    output_lines = eblt_input.to_lines()
    parsed_output = EBLTInput.parse_from_lines(output_lines)

    assert (
        parsed_output == eblt_input
    ), "Test failed: Parsed output does not match the original input object."

    print("Test passed!")
