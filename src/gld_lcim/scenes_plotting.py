import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

DATA_PATH = str(GLOBAL_DIR / "data") + "/"
GENERATED_PATH = str(GLOBAL_DIR / "generated") + "/"
GLD_LCIM_PATH = DATA_PATH + f"gld_lcim/"
STATISTICS_PATH = GLD_LCIM_PATH + "statistics.npy"

FRAME_STEP = 75
FIG_SIZE = (11, 7)


def get_scene_number(filename):
    """
    Returns the scene number from the filename.

    Args:
        filename (str): The filename of the scene

    Returns:
        int: The scene number
    """
    if "Clear" in filename:
        return int(filename.split("Clear")[-1].split(".")[0])
    elif "Overcast" in filename:
        return int(filename.split("Overcast")[-1].split(".")[0])
    else:
        return int(filename.split("e")[-1].split(".")[0])


def show_difference_table(
    statistics,
    paired_keys,
    elements,
    diff_type="mean",
    cell_text_size=80,
    header_text_size=80,
    ):
    """
    Shows a table of the differences between 'Clear' and 'Overcast' conditions.

    Args:
        statistics (dict): The statistics dictionary
        paired_keys (list): The list of paired keys
        elements (list): The list of elements to show
        diff_type (str, optional): The type of difference to show. Defaults to "mean".
        cell_text_size (int, optional): The size of the text in the table cells. Defaults to 80.
        header_text_size (int, optional): The size of the text in the table headers. Defaults to 80.

    Raises:
        ValueError: If the difference type is not 'mean' or 'median'
    """
    if diff_type not in ["mean", "median"]:
        raise ValueError(
            f"❌ Invalid difference type {diff_type}: must be 'mean' or 'median'."
        )

    diff_f = np.mean if diff_type == "mean" else np.median

    # Calculate the differences between means and medians for each element
    differences = {element: {"diff": []} for element in elements}

    for clear_key, overcast_key in paired_keys:
        for element in elements:
            diff = diff_f(statistics[overcast_key][element]) - diff_f(
                statistics[clear_key][element]
            )
            rounded_diff = round(diff, 2)
            differences[element]["diff"].append(rounded_diff)

    # Create the table data
    table_data = []
    for element in elements:
        row = [element]
        row.extend(differences[element]["diff"])
        table_data.append(row)

    # Column labels
    column_labels = ["Metric"]
    column_labels.extend(
        [f"{diff_type.capitalize()} {i+1}" for i in range(len(paired_keys))]
    )

    # Create the table plot
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.axis("tight")
    ax.axis("off")
    t = ax.table(cellText=table_data, colLabels=column_labels, loc="center")

    # Update text size in the table
    for key, cell in t.get_celld().items():
        if key[0] == 0:  # This is a header
            cell.set_fontsize(header_text_size)
        else:
            cell.set_fontsize(cell_text_size)

    plt.title(
        f"{diff_type.capitalize()} Differences Between Clear and Overcast Conditions"
    )

    plot_path = f"{GENERATED_PATH}gld_lcim/statistics_part2_table_{diff_type}.png"
    plt.savefig(plot_path)
    print(f"💾 Table plot saved at {plot_path}")

    plt.show()


def show_plots_part2(statistics):
    """
    Shows the plots for part 2 of the analysis.

    Args:
        statistics (dict): The statistics dictionary
    """
    clear_keys = sorted(
        [key for key in statistics.keys() if key.startswith("P2Clear")],
        key=lambda item: get_scene_number(item),
    )
    overcast_keys = sorted(
        [key for key in statistics.keys() if key.startswith("P2Overcast")],
        key=lambda item: get_scene_number(item),
    )

    # Ensure that we have matching 'Clear' and 'Overcast' scenes
    paired_keys = list(zip(clear_keys, overcast_keys))

    elements = [
        "g_means",
        "diff_g_sums",
        "light_percents",
        "cloud_percents",
        "flow_sums",
    ]
    labels = [
        "Mean Global Brightness",
        "Change in Global Brightness",
        "Light Percentage (%)",
        "Cloud Coverage (%)",
        "Optical Flow Magnitude",
    ]
    colors = ["darkcyan", "darkcyan", "red", "red", "darkgray"]

    # Show and save boxplot
    plt.figure(figsize=FIG_SIZE)
    gs = gridspec.GridSpec(len(elements), 1)

    for i, (element, label, color) in enumerate(zip(elements, labels, colors)):
        ax = plt.subplot(gs[i, 0])

        # We'll create pairs of data for 'Clear' and 'Overcast' to plot side by side
        data_pairs = []
        positions = []
        for j, (clear_key, overcast_key) in enumerate(paired_keys):
            data_pairs.append(statistics[clear_key][element])
            data_pairs.append(statistics[overcast_key][element])
            positions.extend(
                [j * 4 + 1.25, j * 4 + 2.75]
            ) 

        # Plot 'Clear' and 'Overcast' side by side
        ax.boxplot(
            data_pairs,
            positions=positions,
            widths=0.6,
            showfliers=False,
            medianprops={"color": color},
        )

        # Set the primary tick labels for 'Clear' and 'Overcast'
        if i == len(elements) - 1:
            ax.set_xticks(positions)
            ax.set_xticklabels(["Clear", "Overcast"] * len(paired_keys))
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.set_xticks([])

        # Add a secondary axis for scene numbering
        ax.set_title(f"{label} per Scene")
        if i == 0:
            secax = ax.secondary_xaxis("top")
            secax.set_xticks([1.75 + j * 4 for j in range(len(paired_keys))])
            secax.set_xticklabels([f"Scene {j+1}" for j in range(len(paired_keys))])

        # Minor grid for visual separation
        ax.xaxis.grid(True, which="minor", linestyle="", linewidth="0.5", color="gray")
        # Major grid for scene separation
        ax.xaxis.grid(True, which="major", linestyle="", linewidth="0.5", color="black")

    plt.suptitle(f"Video Statistics Analysis (Part 2) with Frame Step {FRAME_STEP}")
    plt.tight_layout()

    plot_path = f"{GENERATED_PATH}gld_lcim/statistics_part2.png"
    plt.savefig(plot_path)
    print(f"💾 Boxplot saved at {plot_path}")

    plt.show()

    # Show and save table plot of differences between 'Clear' and 'Overcast'
    show_difference_table(
        statistics,
        paired_keys,
        elements,
        diff_type="mean",
        cell_text_size=80,
        header_text_size=80,
    )
    show_difference_table(
        statistics,
        paired_keys,
        elements,
        diff_type="median",
        cell_text_size=80,
        header_text_size=80,
    )


def show_plots(statistics: dict, part: int = 1):
    """
    Shows the plots for part 1 to 3 of the analysis.

    Args:
        statistics (dict): The statistics dictionary
        part (int, optional): The part of the analysis to show. Defaults to 1.

    Raises:
        Exception: If the part is not 1, 2 or 3
    """
    if part not in [1, 2, 3]:
        raise Exception(f"❌ Invalid part {part}: must be 1, 2 or 3.")

    if part == 2:
        show_plots_part2(statistics)
        return

    # Sort statistics by taking the ones of the form PX...
    if part == 1:
        filtered_keys = [key for key in statistics.keys() if key.startswith("P1Scene")]
    elif part == 3:
        filtered_keys = [key for key in statistics.keys() if key.startswith("P3Scene")]

    # Sort final statistics by scene number
    statistics = {
        k: statistics[k]
        for k in sorted(filtered_keys, key=lambda item: get_scene_number(item))
    }


    if part == 1:
        # Remove too low values for diff_g_sums and flow_sums in scene 09, which had frozen frames
        statistics['P1Scene09.mp4']['diff_g_sums'] = [
            diff_g_sum
            for diff_g_sum in statistics['P1Scene09.mp4']['diff_g_sums']
            if diff_g_sum > 100_000
        ]
        statistics['P1Scene09.mp4']['flow_sums'] = [
            flow_sum for flow_sum in statistics['P1Scene09.mp4']['flow_sums'] if flow_sum > 0.05
        ]

    # Define the elements to analyze and their corresponding colors
    elements = [
        "g_means",
        "diff_g_sums",
        "light_percents",
        "cloud_percents",
        "flow_sums",
    ]
    labels = [
        "Mean Global Brightness",
        "Change in Global Brightness",
        "Light Percentage (%)",
        "Cloud Coverage (%)",
        "Optical Flow Magnitude",
    ]
    colors = ["darkcyan", "darkcyan", "red", "red", "darkgray"]

    # Create a figure with multiple subplots
    plt.figure(figsize=FIG_SIZE)
    gs = gridspec.GridSpec(len(elements), 1)

    # Plot each element in a separate subplot
    for i, (element, label, color) in enumerate(zip(elements, labels, colors)):
        ax = plt.subplot(gs[i, 0])

        # Create the boxplot for the current element
        ax.boxplot(
            [statistics[scene][element] for scene in statistics.keys()],
            showfliers=False,
            medianprops={"color": color},
        )
        ax.set_title(f"{label} per Scene")
        if i == 0:
            secax = ax.secondary_xaxis("top")
            secax.set_xticks(np.arange(1, len(statistics) + 1))
            secax.set_xticklabels([f"Scene {i}" for i in range(1, len(statistics) + 1)])
        ax.set_xticks([])

    plt.suptitle(f"Video Statistics Analysis (Part {part}) with Frame Step {FRAME_STEP}")
    plt.tight_layout()

    plot_path = f"{GENERATED_PATH}gld_lcim/statistics_part{part}.png"
    plt.savefig(plot_path)
    print(f"💾 Plot saved at {plot_path}")

    plt.show()


if __name__ == "__main__":
    statistics = np.load(STATISTICS_PATH, allow_pickle=True).item()

    for i in range(1, 4):
        show_plots(statistics, part=i)
