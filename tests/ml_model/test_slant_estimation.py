"""Test slant estimation on random ink instances."""

import numpy as np
from matplotlib import pyplot as plt

from constants import TMP_DIR
from ml_model.locals.grpo import _estimate_slant
from schemas.parsed import Parsed


def visualize_ink_with_slant(ink, slant_angle: float, parsed_text: str = "", save_path=None):
    """
    Visualize the original ink with slant estimation overlay.

    Args:
        ink: DigitalInk object
        slant_angle: Estimated slant angle in degrees (90° = vertical)
        parsed_text: Text associated with the ink
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Original ink
    ax1 = axes[0]
    ax1.set_aspect("equal", adjustable="box")
    ax1.invert_yaxis()
    ax1.set_title(f"Original Ink\nText: '{parsed_text}'", fontsize=12)

    for stroke in ink.strokes:
        x = [point.x for point in stroke.points]
        y = [point.y for point in stroke.points]

        if len(stroke.points) == 1:
            ax1.scatter(x, y, s=10, c="k")
        else:
            ax1.plot(x, y, "-k", linewidth=1.5)

    # Plot 2: Ink with slant visualization
    ax2 = axes[1]
    ax2.set_aspect("equal", adjustable="box")
    ax2.invert_yaxis()

    # Describe slant direction
    slant_offset = slant_angle - 90
    if abs(slant_offset) < 2:
        slant_desc = "vertical"
    elif slant_offset > 0:
        slant_desc = f"rightward ({slant_offset:.1f}° from vertical)"
    else:
        slant_desc = f"leftward ({abs(slant_offset):.1f}° from vertical)"

    ax2.set_title(f"Slant Estimation: {slant_angle:.1f}°\n{slant_desc}", fontsize=12)

    # Draw the ink
    for stroke in ink.strokes:
        x = [point.x for point in stroke.points]
        y = [point.y for point in stroke.points]

        if len(stroke.points) == 1:
            ax2.scatter(x, y, s=10, c="k")
        else:
            ax2.plot(x, y, "-k", linewidth=1.5)

    # Calculate bounds for slant lines
    all_x = [point.x for stroke in ink.strokes for point in stroke.points]
    all_y = [point.y for stroke in ink.strokes for point in stroke.points]

    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        center_y = (min_y + max_y) / 2
        height = max_y - min_y

        # Draw slant lines at regular intervals
        num_lines = 7
        for i in range(num_lines):
            x_pos = min_x + (max_x - min_x) * i / (num_lines - 1)

            # Calculate line endpoints based on slant angle
            # slant_angle: 90° = vertical, >90° = rightward slant, <90° = leftward slant
            # Convert to angle from vertical for tan calculation
            angle_from_vertical = slant_angle - 90
            slant_rad = np.radians(angle_from_vertical)

            # Line extends vertically with slant
            dy = height / 2
            dx = dy * np.tan(slant_rad)

            x_start = x_pos - dx
            y_start = center_y - dy
            x_end = x_pos + dx
            y_end = center_y + dy

            ax2.plot(
                [x_start, x_end],
                [y_start, y_end],
                "r--",
                linewidth=1,
                alpha=0.5,
                label="Estimated slant" if i == 0 else "",
            )

        ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", bbox_inches="tight", dpi=150)
        print(f"Saved visualization to: {save_path}")

    return fig


def test_slant_estimation():
    """Test slant estimation on random ink instances."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Number of random samples to test
    num_samples = 5

    print(f"Testing slant estimation on {num_samples} random ink instances...\n")

    for i in range(num_samples):
        print(f"Sample {i + 1}/{num_samples}")

        # Load random parsed instance
        parsed = Parsed.load_random()
        ink = parsed.ink

        # Estimate slant using the grpo module function
        slant = _estimate_slant(ink)

        # Calculate slant offset from vertical for display
        slant_offset = slant - 90

        print(f"  Text: '{parsed.text}'")
        print(f"  Writer: {parsed.writer}")
        print(f"  Estimated slant: {slant:.2f}° ({slant_offset:+.2f}° from vertical)")
        print(f"  Number of strokes: {len(ink.strokes)}")
        print(f"  Total points: {len(ink)}")

        # Create visualization
        save_path = TMP_DIR / f"slant_test_{i + 1}_{parsed.text[:20].replace('/', '_')}.png"
        visualize_ink_with_slant(ink, slant, parsed.text, save_path)
        plt.close()

        print()

    print(f"All visualizations saved to: {TMP_DIR}")


if __name__ == "__main__":
    test_slant_estimation()
