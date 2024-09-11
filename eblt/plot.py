import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Union
from eblt.input import (
    Bend,
    Chicane,
    DriftTube,
    RFCavity,
    WriteBeam,
    ChangeEnergy,
    Wakefield,
    Exit,
    EBLTInput,
)

# Define colors and shapes for different lattice elements
element_styles = {
    Bend: {"color": "red"},
    Chicane: {"color": "purple"},
    DriftTube: {"color": "gray"},
    RFCavity: {"color": "green"},
    WriteBeam: {"color": "orange"},
    ChangeEnergy: {"color": "purple"},
    Wakefield: {"color": "brown"},
    Exit: {"color": "black"},
}


def plot_lattice_lines(
    lattice_lines: List[
        Union[
            Bend, Chicane, DriftTube, RFCavity, WriteBeam, ChangeEnergy, Wakefield, Exit
        ]
    ],
):
    fig, ax = plt.subplots(figsize=(15, 4))
    current_position = 0

    for element in lattice_lines:
        element_type = type(element)
        style = element_styles.get(element_type, {"color": "black"})

        if hasattr(element, "length") and element.length > 0:
            # Determine the height based on angle for Bend and Chicane
            height = 0.1
            if isinstance(element, (Bend, Chicane)):
                height *= max(
                    0.1, abs(element.angle)
                )  # Ensure there's a minimum height

            # Plotting the box for elements with length
            rect = patches.Rectangle(
                (current_position, 0.5 - height / 2),
                element.length,
                height,
                linewidth=1,
                edgecolor="black",
                facecolor=style["color"],
            )
            ax.add_patch(rect)

            # Labeling the element by name if it exists
            if element.name:
                ax.text(
                    current_position + element.length / 2,
                    0.5,
                    element.name,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=10,
                    rotation=90,
                )

            # Update position for the next element
            current_position += element.length
        else:
            # Plotting a vertical line for elements without length
            ax.plot(
                [current_position, current_position],
                [0.5, 1.5],
                color=style["color"],
                linewidth=2,
            )

            # Labeling the element by name if it exists
            if element.name:
                ax.text(
                    current_position,
                    1.5,
                    element.name,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=10,
                    rotation=90,
                )

    # Setting up the plot limits and labels
    ax.set_xlim(0, current_position + 1)  # Add a little space at the end
    ax.set_ylim(0, 2)
    ax.set_xlabel("Distance (m)")
    ax.set_yticks([])  # Hide y-axis ticks
    ax.set_title("Lattice Layout")


# Example usage:
if __name__ == "__main__":
    input_lines = [
        "100 200 0.0d0 1.0d0 1 2 /",
        "1.0 0.1 0.01 0.001 /",
        "0.0 0.0 0.0 0.0 /",
        "0.13 9.27065e+07 511005 -1.0 1.3e+09 /",
        "1.0 10 5 4 0.5 0.5 0.1 0.0 1.01 1 / !name: bend1",
        "0.5 0.5 0.1 0.0 1.01 1 / !name: chicane1",
        "0 0 -2 0.0 0.0 0.0 0.0 / !name: writebeam1",  # Example of an element without length
    ]

    eblt_input = EBLTInput.parse_from_lines(input_lines)
    plot_lattice_lines(eblt_input.lattice_lines)
