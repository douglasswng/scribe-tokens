"""Visualise Scribe tokens as coloured pixels on a discrete grid.

Each atomic Scribe step is rendered as a filled square coloured by its parent
BPE token. Pen-up movement is shown with lower alpha. A zoom inset highlights
one selected token with Freeman-code arrows over its path.

Usage:
    python -m scripts.plot.scribe
"""

import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle

from constants import FIGURES_DIR
from ink_tokeniser.discretes.scribe import STR_TO_COORD
from ink_tokeniser.factory import TokeniserFactory
from ink_tokeniser.id import TokeniserId, TokenType
from ink_tokeniser.tokens import RegularToken, SpecialToken, SpecialTokenType
from schemas.ink import DigitalInk, Point, Stroke

# ── Paths ─────────────────────────────────────────────────────────────────────
SAMPLE_PATH = Path("data/iam/parsed/n07-826z-05.json")
OUTPUT_PATH = FIGURES_DIR / "scribe.pdf"

# ── Tokeniser config ──────────────────────────────────────────────────────────
REPR_ID = TokeniserId(type=TokenType.SCRIBE, delta=32, vocab_size=100000)

# ── Token selection ───────────────────────────────────────────────────────────
_PHI = (1 + 5**0.5) / 2        # ≈ 1.618 (golden ratio for aspect-ratio scoring)
_MIN_CONTRAST_VS_BLACK = 4.5   # WCAG AA minimum contrast for arrow visibility
_MIN_STEPS = 4                 # minimum path positions required for zoom token

# ── Rendering ─────────────────────────────────────────────────────────────────
ARROW_SHRINK_FRAC = 0.0        # fraction of segment to leave clear before arrowhead

# ── Figure layout ─────────────────────────────────────────────────────────────
_FIG_WIDTH = 5.5               # inches; matches NeurIPS \textwidth
_GS_LEFT, _GS_RIGHT = 0.01, 0.99
_GS_TOP, _GS_BOTTOM = 0.95, 0.05
_GS_WSPACE = 0.12

# ── Annotation styling ────────────────────────────────────────────────────────
_ANNOT_COLOR = "black"
_ANNOT_LINEWIDTH = 1.0
_CONNECTOR_COLOR = "0.5"
_CONNECTOR_LINEWIDTH = 0.5


def load_ink_from_json(path: str | Path) -> DigitalInk:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    strokes = []
    for s in data["ink"]["strokes"]:
        points = [Point(x=pt["x"], y=pt["y"]) for pt in s["points"]]
        strokes.append(Stroke(points=points))
    return DigitalInk(strokes=strokes)


Coord = tuple[int, int]
Pixel = tuple[int, int, int]
AtomicToken = RegularToken | SpecialToken


def _append_unique[T](items: list[T], value: T) -> None:
    if not items or items[-1] != value:
        items.append(value)


def _expand_atomic(
    tokeniser, token_ids: list[int]
) -> tuple[list[AtomicToken], list[int], list[AtomicToken]]:
    """Expand BPE tokens into (atomic_tok, bpe_idx) pairs."""
    tokens = tokeniser.convert_ids_to_tokens(token_ids)
    atomic: list[AtomicToken] = []
    atomic_bpe: list[int] = []
    for bpe_idx, tok in enumerate(tokens):
        if isinstance(tok, RegularToken):
            for s in tok.split():
                atomic.append(s)
                atomic_bpe.append(bpe_idx)
        else:
            atomic.append(tok)
            atomic_bpe.append(bpe_idx)
    return atomic, atomic_bpe, tokens


def trace_all(
    tokeniser, token_ids: list[int]
) -> tuple[list[Pixel], list[Pixel], dict[int, list[Coord]]]:
    """Single-pass replay: return pen-down pixels, pen-up pixels, and per-token ordered paths.

    token_paths[bpe_idx] is the ordered sequence of (x, y) positions while the
    pen is down for that token.
    """
    atomic, atomic_bpe, _ = _expand_atomic(tokeniser, token_ids)

    cx, cy = 0, 0
    pen = "down"
    down_pixels: list[Pixel] = []
    up_pixels: list[Pixel] = []
    token_paths: dict[int, list[Coord]] = {}

    def append_pixel(pixels: list[Pixel], x: int, y: int, bpe_idx: int) -> None:
        _append_unique(pixels, (x, y, bpe_idx))

    for tok, bpe in zip(atomic, atomic_bpe):
        if isinstance(tok, SpecialToken):
            if tok.type == SpecialTokenType.START:
                continue
            elif tok.type == SpecialTokenType.END:
                break
            elif tok.type == SpecialTokenType.UP and pen == "down":
                pen = "up"
            elif tok.type == SpecialTokenType.DOWN and pen == "up":
                pen = "down"
        elif isinstance(tok, RegularToken):
            assert isinstance(tok.values, str)
            path = token_paths.setdefault(bpe, [])
            if pen == "down":
                append_pixel(down_pixels, cx, cy, bpe)
                _append_unique(path, (cx, cy))
            else:
                append_pixel(up_pixels, cx, cy, bpe)
            dx, dy = STR_TO_COORD[tok.values]
            cx += dx
            cy += dy
            if pen == "down":
                append_pixel(down_pixels, cx, cy, bpe)
                _append_unique(path, (cx, cy))
            else:
                append_pixel(up_pixels, cx, cy, bpe)

    return down_pixels, up_pixels, token_paths


def _phi_score(path: list[Coord]) -> float:
    """Deviation of the path bounding-box aspect ratio from the golden ratio.

    Landscape tokens (width > height) naturally score better because phi > 1.
    Degenerate zero-height paths get infinity.
    """
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if h == 0:
        return float("inf")
    return abs(w / h - _PHI)


def _contrast_vs_black(rgba: np.ndarray) -> float:
    """WCAG contrast ratio of an sRGB colour against black."""

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    L = 0.2126 * lin(float(rgba[0])) + 0.7152 * lin(float(rgba[1])) + 0.0722 * lin(float(rgba[2]))
    return (L + 0.05) / 0.05


def select_zoom_token(
    token_paths: dict[int, list[Coord]],
    palette: np.ndarray,
) -> int:
    """Pick the token whose bounding box is closest to the golden ratio.

    Hard condition (never relaxed):
      0. Path has at least _MIN_STEPS positions.

    Soft conditions (relaxed in order if needed):
      1. No self-backtracking.
      2. Not overwritten by any later token.
      3. Token colour has sufficient contrast against black (for arrow visibility).
    """
    sorted_bpe_idxs = sorted(token_paths)

    # Build suffix position sets to check condition 2.
    suffix_positions: dict[int, set[Coord]] = {}
    cumulative: set[Coord] = set()
    for bpe_idx in reversed(sorted_bpe_idxs):
        suffix_positions[bpe_idx] = cumulative.copy()
        cumulative.update(token_paths[bpe_idx])

    def long_enough(bpe_idx: int) -> bool:
        return len(token_paths[bpe_idx]) >= _MIN_STEPS

    def no_backtrack(bpe_idx: int) -> bool:
        path = token_paths[bpe_idx]
        return len(set(path)) == len(path)

    def not_overwritten(bpe_idx: int) -> bool:
        return not (set(token_paths[bpe_idx]) & suffix_positions[bpe_idx])

    def good_contrast(bpe_idx: int) -> bool:
        return _contrast_vs_black(color_for_bpe(palette, bpe_idx)) >= _MIN_CONTRAST_VS_BLACK

    base = [k for k in sorted_bpe_idxs if long_enough(k)]
    candidates: list[int] = []
    for filters in [
        lambda k: no_backtrack(k) and not_overwritten(k) and good_contrast(k),
        lambda k: no_backtrack(k) and good_contrast(k),
        lambda k: no_backtrack(k) and not_overwritten(k),
        lambda k: no_backtrack(k),
    ]:
        candidates = [k for k in base if filters(k)]
        if candidates:
            break

    if not candidates:
        candidates = base or sorted_bpe_idxs

    return min(candidates, key=lambda k: _phi_score(token_paths[k]))


def make_color_palette(n: int) -> np.ndarray:
    """Cycle tab10 colours to length n."""
    cmap = plt.get_cmap("tab10")
    base = [cmap(i / 10) for i in range(10)]
    return np.array([base[i % 10] for i in range(n)])


def color_for_bpe(palette: np.ndarray, bpe: int, alpha: float = 1.0) -> np.ndarray:
    c = palette[bpe % len(palette)].copy()
    c[3] = alpha
    return c


def draw_pixels(
    ax: Axes,
    pixels: list[Pixel],
    palette: np.ndarray,
    alpha: float = 1.0,
    zorder: int = 2,
) -> None:
    if not pixels:
        return
    rects = [Rectangle((x - 0.5, y - 0.5), 1, 1) for x, y, _ in pixels]
    colors = [color_for_bpe(palette, bpe, alpha) for _, _, bpe in pixels]
    ax.add_collection(PatchCollection(rects, facecolors=colors, edgecolors="none", zorder=zorder))


def add_path_arrows(
    ax: Axes,
    path: list[Coord],
    color: str = "black",
    linewidth: float = 1.0,
    head_scale: float = 6.0,
    shrink_frac: float = ARROW_SHRINK_FRAC,
    zorder: int = 5,
) -> None:
    """Draw Freeman-chain arrows along consecutive positions.

    Arrow endpoints are offset by shrink_frac of the segment length in data
    space, so the bodies are always visible regardless of scale.
    """
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        if (x0, y0) == (x1, y1):
            continue
        dx, dy = x1 - x0, y1 - y0
        sx = x0 + shrink_frac * dx
        sy = y0 + shrink_frac * dy
        ex = x1 - shrink_frac * dx
        ey = y1 - shrink_frac * dy
        ax.add_patch(
            FancyArrowPatch(
                (sx, sy),
                (ex, ey),
                arrowstyle="->",
                mutation_scale=head_scale,
                lw=linewidth,
                color=color,
                shrinkA=0,
                shrinkB=0,
                zorder=zorder,
            )
        )


def _draw_zoom_panel(
    fig: Figure,
    ax_main: Axes,
    ax_zoom: Axes,
    zoom_bpe_idx: int,
    token_paths: dict[int, list[Coord]],
    down_pixels: list[Pixel],
    up_pixels: list[Pixel],
    palette: np.ndarray,
    connect_side: Literal["left", "right"],
) -> None:
    """Draw a single zoom panel and its connecting lines.

    The zoom panel shows a true magnification of the main view content,
    with Freeman-chain arrows overlaid on the featured token's path.
    """
    zoom_path = token_paths[zoom_bpe_idx]

    # pad=2: gives a clean half-pixel margin outside the outermost pixel edges
    pad = 2
    zx_min = min(p[0] for p in zoom_path) - pad
    zx_max = max(p[0] for p in zoom_path) + pad
    zy_min = min(p[1] for p in zoom_path) - pad
    zy_max = max(p[1] for p in zoom_path) + pad

    # ── Focus box on main panel ───────────────────────────────────────────────
    ax_main.add_patch(Rectangle(
        (zx_min, zy_min),
        zx_max - zx_min,
        zy_max - zy_min,
        linewidth=_ANNOT_LINEWIDTH,
        edgecolor=_ANNOT_COLOR,
        facecolor="none",
        zorder=6,
    ))

    # ── Zoom panel: same rendering as main view ───────────────────────────────
    draw_pixels(ax_zoom, down_pixels, palette, alpha=1.0, zorder=2)
    draw_pixels(ax_zoom, up_pixels, palette, alpha=0.1, zorder=1)
    add_path_arrows(ax_zoom, zoom_path, color="black", linewidth=1.0, head_scale=7.0)

    ax_zoom.set_xlim(zx_min, zx_max)
    ax_zoom.set_ylim(zy_min, zy_max)
    ax_zoom.invert_yaxis()
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    # Replace spine Line2Ds with a single closed Rectangle patch for uniform edge weight.
    for spine in ax_zoom.spines.values():
        spine.set_visible(False)
    ax_zoom.add_patch(Rectangle(
        (0, 0),
        1,
        1,
        transform=ax_zoom.transAxes,
        linewidth=_ANNOT_LINEWIDTH,
        edgecolor=_ANNOT_COLOR,
        facecolor="none",
        clip_on=False,
        zorder=10,
    ))

    # ── Connecting lines ──────────────────────────────────────────────────────
    # Connect the facing-side corners: right edge for left inset, left for right.
    corner_x = zx_max if connect_side == "left" else zx_min
    for y_val in (zy_min, zy_max):
        fig.add_artist(ConnectionPatch(
            xyA=(corner_x, y_val),
            coordsA="data",
            axesA=ax_main,
            xyB=(corner_x, y_val),
            coordsB="data",
            axesB=ax_zoom,
            linewidth=_CONNECTOR_LINEWIDTH,
            color=_CONNECTOR_COLOR,
            linestyle="--",
            clip_on=False,
            zorder=6,
        ))


def _compute_figsize(
    all_x: list[int],
    all_y: list[int],
    width_ratios: list[int],
) -> tuple[float, float]:
    """Derive figure height so the main axis renders at the ink's native aspect ratio.

    Given fixed GridSpec parameters and a target figure width, the main axis
    width is determined analytically. The height is then chosen so that
    aspect='equal' on that axis displays the ink without distortion.
    """
    ink_aspect = (max(all_x) - min(all_x)) / max(max(all_y) - min(all_y), 1e-6)

    subplot_width = _FIG_WIDTH * (_GS_RIGHT - _GS_LEFT)
    n_cols = len(width_ratios)
    sum_ratios = sum(width_ratios)
    avg_ratio = sum_ratios / n_cols
    # subplot_width = unit * (sum_ratios + (n_cols-1) * wspace * avg_ratio)
    unit = subplot_width / (sum_ratios + (n_cols - 1) * _GS_WSPACE * avg_ratio)
    main_ax_width = max(width_ratios) * unit

    main_ax_height = main_ax_width / ink_aspect
    fig_height = main_ax_height / (_GS_TOP - _GS_BOTTOM)
    return _FIG_WIDTH, fig_height


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ink = load_ink_from_json(SAMPLE_PATH)
    tokeniser = TokeniserFactory.create(REPR_ID)

    token_ids = tokeniser.encode(ink)
    n_bpe = len(token_ids)
    _, _, tokens = _expand_atomic(tokeniser, token_ids)
    print(f"Sample: {SAMPLE_PATH}")
    print(f"BPE tokens: {n_bpe}")

    down_pixels, up_pixels, token_paths = trace_all(tokeniser, token_ids)
    palette = make_color_palette(n_bpe)

    zoom_bpe_idx = select_zoom_token(token_paths, palette)

    zoom_token = tokens[zoom_bpe_idx]
    assert isinstance(zoom_token, RegularToken)
    print(f"Zoom token {zoom_bpe_idx}: {zoom_token.values!r}")

    # ── Determine inset side based on token position ──────────────────────────
    all_pixels = down_pixels + up_pixels
    all_x = [p[0] for p in all_pixels]
    all_y = [p[1] for p in all_pixels]
    canvas_cx = (min(all_x) + max(all_x)) / 2
    zoom_path = token_paths[zoom_bpe_idx]
    token_cx = (min(p[0] for p in zoom_path) + max(p[0] for p in zoom_path)) / 2
    zoom_on_left = token_cx < canvas_cx
    connect_side: Literal["left", "right"] = "left" if zoom_on_left else "right"

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
        }
    )

    # ── Layout: [zoom] [main] or [main] [zoom] ────────────────────────────────
    width_ratios = [1, 5] if zoom_on_left else [5, 1]
    figsize = _compute_figsize(all_x, all_y, width_ratios)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=width_ratios,
        wspace=_GS_WSPACE,
        left=_GS_LEFT,
        right=_GS_RIGHT,
        top=_GS_TOP,
        bottom=_GS_BOTTOM,
    )
    if zoom_on_left:
        ax_zoom = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1])
    else:
        ax = fig.add_subplot(gs[0])
        ax_zoom = fig.add_subplot(gs[1])

    ax.set_aspect("equal", adjustable="box")
    ax_zoom.set_aspect("equal", adjustable="box")

    # ── Main view ─────────────────────────────────────────────────────────────
    draw_pixels(ax, down_pixels, palette, alpha=1.0, zorder=2)
    draw_pixels(ax, up_pixels, palette, alpha=0.1, zorder=1)

    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Zoom panel ────────────────────────────────────────────────────────────
    _draw_zoom_panel(
        fig=fig,
        ax_main=ax,
        ax_zoom=ax_zoom,
        zoom_bpe_idx=zoom_bpe_idx,
        token_paths=token_paths,
        down_pixels=down_pixels,
        up_pixels=up_pixels,
        palette=palette,
        connect_side=connect_side,
    )

    fig.savefig(OUTPUT_PATH, dpi=300)
    print(f"Saved {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
