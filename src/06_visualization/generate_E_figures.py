"""
generate_E_figures.py

Generates 4 separate figure files (E1.png to E4.png).
Each figure contains a single row of 3 human avatars sharing the exact same color.
Each of the 4 output figures utilizes a distinct color from the palette.
Avatars are spaced closely together.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

# Dynamically resolve to the base directory (assuming script is 2 folders deep)
BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "outputs" / "figures"

PALETTE = [
    "#E58A8A",  # Soft Red/Pink
    "#8FCE8A",  # Mint Green
    "#F3D266",  # Warm Yellow
    "#6AA2DE",  # Soft Blue
    "#B9A0CD",  # Lavender/Purple
    "#F1B495",  # Peachy Orange
]

# ==========================================
# HELPER FUNCTION
# ==========================================

def draw_avatar(ax, x, y, color):
    """
    Draws a minimalist human bust with a clean white outline.
    """
    verts = [
        (x - 24, y),
        (x - 24, y - 35),
        (x + 24, y - 35),
        (x + 24, y),
        (x - 24, y),
    ]

    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]

    shoulders = patches.PathPatch(
        MplPath(verts, codes),
        facecolor=color,
        edgecolor="white",
        lw=3.5,
        zorder=y,
    )
    ax.add_patch(shoulders)

    head = patches.Circle(
        (x, y - 34),
        radius=13,
        facecolor=color,
        edgecolor="white",
        lw=3.5,
        zorder=y,
    )
    ax.add_patch(head)


# ==========================================
# MAIN ROUTINE
# ==========================================

def main():
    # Ensure the output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Select 4 distinct colors from the palette for the 4 images
    figure_colors = [
        PALETTE[0],  # E1: Soft Red/Pink
        PALETTE[1],  # E2: Mint Green
        PALETTE[3],  # E3: Soft Blue
        PALETTE[4],  # E4: Lavender/Purple
    ]

    # X coordinates adjusted to bring the avatars closer together 
    # (Gap reduced from 100 to 60 units)
    x_positions = [140, 200, 260]
    y_position = 120

    for i, color in enumerate(figure_colors, start=1):
        # Create a wider, shorter canvas to frame the single row nicely
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.set_xlim(40, 360)
        ax.set_ylim(160, 40)
        ax.set_aspect("equal")
        ax.axis("off")

        # Render the 3 avatars in a row using the SAME color for this specific figure
        for x in x_positions:
            draw_avatar(ax, x, y_position, color)

        # Export
        out_path = OUT_DIR / f"E{i}.png"
        fig.savefig(
            out_path,
            dpi=600,
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
        )

        plt.close(fig)
        print(f"[*] Successfully generated {out_path.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()