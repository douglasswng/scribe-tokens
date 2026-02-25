"""Generate a grid figure of handwriting samples from HTG models.

Each method has two sub-rows (No PT and +PT), matching the table format in
results.py. Columns are text prompts. All generations use temperature=1.0.

Usage:
    python -m scripts.plot.htg
"""

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from constants import DATASET, FIGURES_DIR
from ink_repr.id import ReprId, VectorReprId
from ink_tokeniser.id import TokeniserId
from ml_model.factory import ModelFactory
from ml_model.id import ModelId, Task
from schemas.ink import DigitalInk, Point, Stroke
from schemas.instance import Instance
from utils.set_random_seed import set_random_seed

assert DATASET == "iam", f"Expected dataset 'iam', got '{DATASET}'"

OUTPUT_PATH = FIGURES_DIR / "htg.pdf"

PROMPTS = [
    "hello world",
    "the quick brown",
]
NUM_PROMPTS = len(PROMPTS)

METHODS: list[tuple[str, ReprId]] = [
    ("Point-5", VectorReprId.create_point5()),
    ("RelTokens", TokeniserId.create_rel()),
    ("TextTokens", TokeniserId.create_text()),
    ("ScribeTokens (Ours)", TokeniserId.create_scribe()),
]

TRAINING_TYPES: list[tuple[str, Task]] = [
    ("", Task.HTG),
    ("+PT", Task.HTG_SFT),
]
_PT_INDEX = 1  # index of +PT in TRAINING_TYPES (used to determine clipping bounds)

TEMPERATURE = 1.0

# ── Figure layout (inches) ───────────────────────────────────────────────────
_FIG_WIDTH = 5.5
_METHOD_COL_IN = 1.15  # space for method name column (left-aligned)
_GAP_COL_IN = 0.10  # space between method and training columns
_TRAINING_COL_IN = 0.22  # space for training label column
_GAP_RIGHT_IN = 0.15  # space between training column and ink plots
_LEFT_IN = _METHOD_COL_IN + _GAP_COL_IN + _TRAINING_COL_IN + _GAP_RIGHT_IN
_RIGHT_IN = 0.05
_TOP_IN = 0.25  # space for column titles
_BOTTOM_IN = 0.05
_WSPACE = 0.0
_HSPACE_WITHIN = 0.05  # vertical space between sub-rows of the same method
_HSPACE_BETWEEN = 0.25  # vertical space between method groups
_PAD_FRAC = 0.05  # fractional padding around ink bounding boxes


def _get_bounding_box(ink: DigitalInk) -> tuple[float, float, float, float]:
    """Compute (xmin, xmax, ymin, ymax) for an ink."""
    all_x, all_y = [], []
    for stroke in ink.strokes:
        for pt in stroke.points:
            all_x.append(pt.x)
            all_y.append(pt.y)
    return min(all_x), max(all_x), min(all_y), max(all_y)


def _clip_ink(ink: DigitalInk, x_max: float) -> DigitalInk:
    """Remove points that extend beyond x_max, keeping partial strokes."""
    clipped_strokes: list[Stroke] = []
    for stroke in ink.strokes:
        clipped_points: list[Point] = []
        for pt in stroke.points:
            if pt.x > x_max:
                break
            clipped_points.append(pt)
        if clipped_points:
            clipped_strokes.append(Stroke(points=clipped_points))
    return DigitalInk(strokes=clipped_strokes) if clipped_strokes else ink


def _compute_limits(
    all_inks: list[list[DigitalInk]],
    ncols: int,
    num_training: int,
) -> tuple[float, float, list[list[tuple[float, float]]]]:
    """Compute global x-span, global y-span, and per-cell y-centers.

    x-span and y-span are derived only from +PT rows so that degenerate
    No PT generations are clipped to the same extent.

    Returns:
        x_span: global x-span with padding (from +PT rows only)
        y_span: global y-span with padding (from +PT rows only)
        cell_centers: per-cell (x_start, y_center) for all rows
    """
    nrows = len(all_inks)

    bboxes = [[_get_bounding_box(all_inks[r][c]) for c in range(ncols)] for r in range(nrows)]

    # Compute spans only from +PT rows
    max_x_span = 0.0
    max_y_span = 0.0
    for row_idx in range(nrows):
        if row_idx % num_training != _PT_INDEX:
            continue
        for col_idx in range(ncols):
            xmin, xmax, ymin, ymax = bboxes[row_idx][col_idx]
            max_x_span = max(max_x_span, xmax - xmin)
            max_y_span = max(max_y_span, ymax - ymin)

    x_span = max_x_span * (1 + 2 * _PAD_FRAC)
    y_span = max_y_span * (1 + 2 * _PAD_FRAC)

    cell_centers: list[list[tuple[float, float]]] = []
    for row_idx in range(nrows):
        row_centers = []
        for col_idx in range(ncols):
            xmin, _, ymin, ymax = bboxes[row_idx][col_idx]
            row_centers.append((xmin, (ymin + ymax) / 2))
        cell_centers.append(row_centers)

    return x_span, y_span, cell_centers


def _compute_figsize(
    x_span: float,
    y_span: float,
    num_methods: int,
    num_training: int,
    ncols: int,
) -> tuple[float, float]:
    """Derive figure height from the shared spans, accounting for method group spacing."""
    aspect = x_span / max(y_span, 1e-6)

    subplot_region_width = _FIG_WIDTH - _LEFT_IN - _RIGHT_IN
    col_width = subplot_region_width / ncols  # wspace=0
    row_height = col_width / aspect

    total_rows = num_methods * num_training
    within_gaps = num_methods * (num_training - 1)
    between_gaps = num_methods - 1
    subplot_region_height = (
        total_rows * row_height
        + within_gaps * _HSPACE_WITHIN * row_height
        + between_gaps * _HSPACE_BETWEEN * row_height
    )
    fig_height = subplot_region_height + _TOP_IN + _BOTTOM_IN
    return _FIG_WIDTH, fig_height


def _build_height_ratios(num_methods: int, num_training: int) -> list[float]:
    """Build GridSpec height_ratios with spacing rows between method groups."""
    ratios: list[float] = []
    for i in range(num_methods):
        for j in range(num_training):
            ratios.append(1.0)
            if j < num_training - 1:
                ratios.append(_HSPACE_WITHIN)
        if i < num_methods - 1:
            ratios.append(_HSPACE_BETWEEN)
    return ratios


def _draw_ink(ax: Axes, ink: DigitalInk) -> None:
    """Draw digital ink strokes onto an axes."""
    for stroke in ink.strokes:
        x = [p.x for p in stroke.points]
        y = [p.y for p in stroke.points]
        if len(stroke.points) == 1:
            ax.scatter(x, y, s=1.0, c="k")
        else:
            ax.plot(x, y, "-k", linewidth=0.7)


def generate_ink(model, repr_id: ReprId, text: str) -> DigitalInk:
    """Generate a single ink sample from text."""
    instance = Instance.from_text(text, repr_id).to_device()
    inks = model.batch_generate_inks([instance], temperature=TEMPERATURE, num_generations=1)
    return inks[0][0]


def main() -> None:
    set_random_seed()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
        }
    )

    num_methods = len(METHODS)
    num_training = len(TRAINING_TYPES)
    num_cols = NUM_PROMPTS

    # ── Pass 1: generate all inks ─────────────────────────────────────────────
    # all_inks[flat_row][col] where flat_row = method_idx * num_training + training_idx
    all_inks: list[list[DigitalInk]] = []
    for method_name, repr_id in METHODS:
        for training_label, task in TRAINING_TYPES:
            label = f"{method_name} {training_label}".strip()
            print(f"Loading {label}...")
            model_id = ModelId(task=task, repr_id=repr_id)
            model = ModelFactory.load_pretrained(model_id)
            model.eval()

            row_inks = []
            for prompt in PROMPTS:
                print(f"  Generating: {prompt!r}")
                ink = generate_ink(model.local_model, repr_id, prompt)
                row_inks.append(ink)
            all_inks.append(row_inks)

            del model
            torch.cuda.empty_cache()

    # ── Clip No PT inks to the +PT x-extent ───────────────────────────────────
    for method_idx in range(num_methods):
        pt_row = method_idx * num_training + _PT_INDEX
        for training_idx in range(num_training):
            if training_idx == _PT_INDEX:
                continue
            flat_row = method_idx * num_training + training_idx
            for col_idx in range(num_cols):
                _, pt_xmax, _, _ = _get_bounding_box(all_inks[pt_row][col_idx])
                all_inks[flat_row][col_idx] = _clip_ink(
                    all_inks[flat_row][col_idx],
                    pt_xmax,
                )

    # ── Pass 2: compute layout and render ─────────────────────────────────────
    x_span, y_span, cell_centers = _compute_limits(all_inks, num_cols, num_training)
    figsize = _compute_figsize(x_span, y_span, num_methods, num_training, num_cols)

    height_ratios = _build_height_ratios(num_methods, num_training)
    gs_rows = len(height_ratios)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        gs_rows,
        num_cols,
        figure=fig,
        height_ratios=height_ratios,
        left=_LEFT_IN / figsize[0],
        right=1 - _RIGHT_IN / figsize[0],
        top=1 - _TOP_IN / figsize[1],
        bottom=_BOTTOM_IN / figsize[1],
        hspace=0,
        wspace=_WSPACE,
    )

    x_pad = x_span * _PAD_FRAC / (1 + 2 * _PAD_FRAC)

    # Label x-positions in figure fraction
    method_x = 0.02 / figsize[0]  # left-aligned with small indent
    training_x = (_METHOD_COL_IN + _GAP_COL_IN + _TRAINING_COL_IN / 2) / figsize[
        0
    ]  # centered in training column

    flat_row = 0
    gs_row = 0
    for method_idx, (method_name, _) in enumerate(METHODS):
        for training_idx, (training_label, _) in enumerate(TRAINING_TYPES):
            for col_idx in range(num_cols):
                ax = fig.add_subplot(gs[gs_row, col_idx])
                ink = all_inks[flat_row][col_idx]
                _draw_ink(ax, ink)

                x_start, yc = cell_centers[flat_row][col_idx]
                ax.set_xlim(x_start - x_pad, x_start - x_pad + x_span)
                ax.set_ylim(yc - y_span / 2, yc + y_span / 2)
                ax.invert_yaxis()
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

                if flat_row == 0:
                    ax.set_title(f'"{PROMPTS[col_idx]}"', fontsize=8, pad=4)

            # Row labels — method name (left-aligned) and training type (centered)
            ax_first = fig.axes[-num_cols]
            bbox = ax_first.get_position()
            row_y = (bbox.y0 + bbox.y1) / 2

            if training_idx == 0:
                fig.text(
                    method_x,
                    row_y,
                    method_name,
                    fontsize=8,
                    ha="left",
                    va="center",
                )
            fig.text(
                training_x,
                row_y,
                training_label,
                fontsize=8,
                ha="center",
                va="center",
            )

            flat_row += 1
            gs_row += 1

            # Skip the within-group spacing row
            if training_idx < num_training - 1:
                gs_row += 1

        # Skip the between-group spacing row
        if method_idx < num_methods - 1:
            gs_row += 1

    fig.savefig(OUTPUT_PATH, dpi=300)
    print(f"Saved {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
