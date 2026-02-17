"""Attention analysis: ink attention table for ScribeTokens and TextTokens.

For each model, generates a static PDF with one row per decoded character.
Each row has 3 columns:
1) completion so far, with each character colored by its attention score,
2) target character for this step,
3) reconstructed ink colored by per-token attention for this step.

Usage:
    python -m scripts.eval.attention
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from scipy.signal import savgol_filter

from constants import DATASET, FIGURES_DIR
from ink_tokeniser.discretes.scribe import STR_TO_COORD, ScribeTokeniser
from ink_tokeniser.discretes.text import SEP, TextTokeniser
from ink_tokeniser.factory import TokeniserFactory
from ink_tokeniser.id import TokeniserId
from ink_tokeniser.tokeniser import Tokeniser
from ink_tokeniser.tokens import RegularToken, SpecialToken, SpecialTokenType
from ml_model.factory import ModelFactory
from ml_model.id import ModelId, Task
from ml_model.locals.htr import HTRModel
from ml_model.modules.mha import MultiHeadAttention
from schemas.ink import DigitalInk
from schemas.instance import Instance
from schemas.parsed import Parsed

assert DATASET == "iam"

SAMPLE_PATH = "data/iam/parsed/a01-007z-07.json"
OUTPUT_HTR_SCRIBE = FIGURES_DIR / "attn_htr_scribe.pdf"
OUTPUT_HTR_TEXT = FIGURES_DIR / "attn_htr_text.pdf"
OUTPUT_HTR_SFT_SCRIBE = FIGURES_DIR / "attn_htr_sft_scribe.pdf"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
SPACE_TOKEN = "␣"


class AttentionCapture:
    """Accumulates attention weights from patched MHA layers during decode steps."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.step_weights: list[list[torch.Tensor]] = []
        self._buf: list[torch.Tensor | None] = [None] * num_layers

    def _flush_buf(self) -> None:
        assert all(w is not None for w in self._buf)
        self.step_weights.append([w for w in self._buf if w is not None])
        self._buf = [None] * self.num_layers

    def record(self, layer_idx: int, weights: torch.Tensor) -> None:
        if layer_idx == 0 and any(w is not None for w in self._buf):
            self._flush_buf()
        self._buf[layer_idx] = weights

    def flush(self) -> None:
        if any(w is not None for w in self._buf):
            self._flush_buf()


def _make_patched_forward(mha: MultiHeadAttention, layer_idx: int, capture: AttentionCapture):
    def forward(
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        start_pos: int | torch.Tensor = 0,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        q = mha.q_proj(x)
        k = mha.k_proj(x)
        v = mha.v_proj(x)
        q = q.view(batch_size, seq_len, mha.n_heads, mha.head_dim)
        k = k.view(batch_size, seq_len, mha.n_heads, mha.head_dim)
        v = v.view(batch_size, seq_len, mha.n_heads, mha.head_dim)
        q = mha.rope.apply_rotary_emb(q, start_pos)
        k = mha.rope.apply_rotary_emb(k, start_pos)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)
        scores = torch.matmul(q, k.transpose(-2, -1)) * mha.scale
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        if seq_len == 1:
            capture.record(layer_idx, weights.detach().cpu())
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, mha.d_model)
        out = mha.out_proj(out)
        return out, new_kv_cache

    return forward


def patch_model(model: HTRModel, capture: AttentionCapture) -> None:
    for layer_idx, layer in enumerate(model._decoder.layers):
        mha = layer.self_attn
        mha.forward = _make_patched_forward(mha, layer_idx, capture)  # type: ignore[assignment]


def rollout_attention(step_weights: list[list[torch.Tensor]]) -> list[np.ndarray]:
    """Attention rollout across decoder layers (Abnar & Zuidema, 2020).

    During cached autoregressive decoding we only have the query token's
    attention row at each layer, not the full NxN matrix.  We approximate
    rollout by iteratively blending each layer's head-averaged attention
    with the previous rollout via the residual connection:

        r_0 = a_0
        r_l = 0.5 * a_l + 0.5 * r_{l-1}   (renormalised)

    Returns list of 1D arrays, one per step, shape (kv_len,).
    """
    distributions: list[np.ndarray] = []
    for layer_ws in step_weights:
        per_layer = [w.squeeze(0).squeeze(1).mean(dim=0) for w in layer_ws]
        rollout = per_layer[0]
        for attn in per_layer[1:]:
            rollout = 0.5 * attn + 0.5 * rollout
            rollout = rollout / rollout.sum()
        distributions.append(rollout.numpy())
    return distributions


def _split_bpe_tokens(tokeniser: Tokeniser, token_ids: list[int]) -> tuple[list[object], list[int]]:
    """Split BPE tokens back to atomic, tracking which BPE index each came from."""
    tokens = tokeniser.convert_ids_to_tokens(token_ids)

    atomic: list[object] = []
    atomic_to_bpe: list[int] = []
    for bpe_idx, tok in enumerate(tokens):
        if isinstance(tok, RegularToken):
            for s in tok.split():
                atomic.append(s)
                atomic_to_bpe.append(bpe_idx)
        else:
            atomic.append(tok)
            atomic_to_bpe.append(bpe_idx)
    return atomic, atomic_to_bpe


def _trace_scribe(
    atomic: list[object], atomic_to_bpe: list[int], delta: float
) -> tuple[list[np.ndarray], list[list[list[int]]]]:
    """Replay ScribeTokeniser.detokenise with BPE index tracking.

    Returns:
        strokes: list of (N, 2) arrays (scaled coordinates).
        point_bpe: parallel structure; point_bpe[s][p] = list of BPE indices
                   that produced point p in stroke s.
    """
    current = np.array([0.0, 0.0])
    strokes: list[list[np.ndarray]] = [[current.copy()]]
    point_bpe: list[list[list[int]]] = [[[]]]  # initial point has no token
    pen = "down"

    for i, tok in enumerate(atomic):
        bpe = atomic_to_bpe[i]
        if isinstance(tok, SpecialToken):
            if tok.type == SpecialTokenType.START:
                continue
            elif tok.type == SpecialTokenType.END:
                break
            elif tok.type == SpecialTokenType.UP and pen == "down":
                pen = "up"
            elif tok.type == SpecialTokenType.DOWN and pen == "up":
                pen = "down"
                strokes.append([current.copy()])
                point_bpe.append([[]])
        elif isinstance(tok, RegularToken):
            assert isinstance(tok.values, str)
            dx, dy = STR_TO_COORD[tok.values]
            current = current + np.array([float(dx), float(dy)])
            if pen == "down":
                strokes[-1].append(current.copy())
                point_bpe[-1].append([bpe])

    # Scale and convert to arrays
    scaled = [np.array(s) * delta for s in strokes]
    return scaled, point_bpe


def _trace_text(
    atomic: list[object], atomic_to_bpe: list[int], delta: float
) -> tuple[list[np.ndarray], list[list[list[int]]]]:
    """Replay TextTokeniser.detokenise with BPE index tracking.

    Returns same format as _trace_scribe.
    """
    # Group consecutive regular tokens into (string, bpe_indices) pairs,
    # keeping special tokens separate.
    hybrid: list[tuple[str, str | SpecialToken, list[int]]] = []
    i = 0
    while i < len(atomic):
        tok = atomic[i]
        if isinstance(tok, RegularToken):
            chars: list[str] = []
            bpes: list[int] = []
            while i < len(atomic) and isinstance(atomic[i], RegularToken):
                reg = atomic[i]
                assert isinstance(reg, RegularToken)
                assert isinstance(reg.values, str)
                chars.append(reg.values)
                bpes.append(atomic_to_bpe[i])
                i += 1
            hybrid.append(("text", "".join(chars), bpes))
        else:
            assert isinstance(tok, SpecialToken)
            hybrid.append(("special", tok, [atomic_to_bpe[i]]))
            i += 1

    # Walk through hybrid repr, parsing points and tracking BPE
    current = np.array([0.0, 0.0])
    stroke_pts: list[np.ndarray] = [current.copy()]
    stroke_bpe: list[list[int]] = [[]]
    strokes: list[list[np.ndarray]] = []
    all_point_bpe: list[list[list[int]]] = []

    for kind, val, bpes in hybrid:
        if kind == "special":
            assert isinstance(val, SpecialToken)
            if val.type == SpecialTokenType.START:
                continue
            elif val.type == SpecialTokenType.END:
                break
            elif val.type == SpecialTokenType.UP:
                strokes.append(stroke_pts)
                all_point_bpe.append(stroke_bpe)
                stroke_pts = []
                stroke_bpe = []
        else:
            assert isinstance(val, str)
            # Parse string into coordinate pairs, tracking char→BPE
            parts = val.split(SEP)
            # Track cumulative char position to map parts back to chars
            char_pos = 0
            for pi in range(0, len(parts) - 1, 2):
                try:
                    x = int(parts[pi])
                    y = int(parts[pi + 1])
                except (ValueError, IndexError):
                    # Skip malformed coordinates
                    char_pos += len(parts[pi]) + 1  # +1 for SEP
                    if pi + 1 < len(parts):
                        char_pos += len(parts[pi + 1]) + 1
                    continue

                # Chars for this point: x_str + SEP + y_str (+ optional trailing SEP)
                start = char_pos
                char_pos += len(parts[pi]) + 1  # x + SEP
                char_pos += len(parts[pi + 1])  # y
                end = char_pos
                if pi + 2 < len(parts) - 1:
                    char_pos += 1  # trailing SEP between points

                # Collect unique BPE indices for this point's chars
                assert start >= 0 and end <= len(bpes), (
                    f"Char range [{start}:{end}) out of bounds for "
                    f"bpes of length {len(bpes)} "
                    f"(string={val!r}, parts={parts}, pi={pi})"
                )
                point_bpes = list(set(bpes[start:end]))

                current = current + np.array([float(x), float(y)])
                stroke_pts.append(current.copy())
                stroke_bpe.append(point_bpes)

    # Handle any remaining stroke
    if stroke_pts:
        strokes.append(stroke_pts)
        all_point_bpe.append(stroke_bpe)

    scaled = [np.array(s) * delta for s in strokes]
    return scaled, all_point_bpe


def _validate_trace(
    strokes: list[np.ndarray],
    point_bpe: list[list[list[int]]],
    ref_ink: DigitalInk,
    ink_len: int,
    label: str,
) -> None:
    """Validate that the traced ink matches the reference decode and structure is consistent."""

    # 1. Stroke count must match
    assert len(strokes) == len(ref_ink.strokes), (
        f"[{label}] Stroke count mismatch: traced {len(strokes)} vs decoded {len(ref_ink.strokes)}"
    )

    # 2. Point count per stroke must match
    for s_idx, (traced_stroke, ref_stroke) in enumerate(zip(strokes, ref_ink.strokes)):
        assert len(traced_stroke) == len(ref_stroke.points), (
            f"[{label}] Stroke {s_idx} point count mismatch: "
            f"traced {len(traced_stroke)} vs decoded {len(ref_stroke.points)}"
        )

    # 3. point_bpe must be parallel to strokes
    assert len(point_bpe) == len(strokes), (
        f"[{label}] point_bpe has {len(point_bpe)} strokes but traced ink has {len(strokes)}"
    )
    for s_idx in range(len(strokes)):
        assert len(point_bpe[s_idx]) == len(strokes[s_idx]), (
            f"[{label}] Stroke {s_idx}: point_bpe has {len(point_bpe[s_idx])} entries "
            f"but stroke has {len(strokes[s_idx])} points"
        )

    # 4. All BPE indices must be in valid range [0, ink_len)
    for s_idx, stroke_bpes in enumerate(point_bpe):
        for p_idx, bpe_list in enumerate(stroke_bpes):
            for bpe in bpe_list:
                assert 0 <= bpe < ink_len, (
                    f"[{label}] Stroke {s_idx}, point {p_idx}: "
                    f"BPE index {bpe} out of range [0, {ink_len})"
                )

    # 5. Coordinates must match (within floating point tolerance)
    for s_idx, (traced_stroke, ref_stroke) in enumerate(zip(strokes, ref_ink.strokes)):
        for p_idx, (traced_pt, ref_pt) in enumerate(zip(traced_stroke, ref_stroke.points)):
            ref_xy = np.array([float(ref_pt.x), float(ref_pt.y)])
            assert np.allclose(traced_pt, ref_xy, atol=1e-6), (
                f"[{label}] Stroke {s_idx}, point {p_idx}: "
                f"coordinate mismatch traced={traced_pt} vs decoded={ref_xy}"
            )

    print(
        f"  [{label}] Validation passed: {len(strokes)} strokes, "
        f"{sum(len(s) for s in strokes)} total points"
    )


def build_ink_and_mapping(
    repr_id: TokeniserId, ink_token_ids: list[int]
) -> tuple[list[np.ndarray], list[list[list[int]]]]:
    """Reconstruct ink and build per-point BPE token index mapping.

    Returns:
        strokes: list of (N, 2) coordinate arrays for drawing.
        point_bpe: point_bpe[stroke][point] = list of BPE token indices.
    """
    tokeniser = TokeniserFactory.create(repr_id)
    atomic, atomic_to_bpe = _split_bpe_tokens(tokeniser, ink_token_ids)

    discrete = tokeniser._discrete_tokeniser
    delta: float = getattr(tokeniser._preprocessor, "_delta")

    if isinstance(discrete, ScribeTokeniser):
        strokes, point_bpe = _trace_scribe(atomic, atomic_to_bpe, delta)
    elif isinstance(discrete, TextTokeniser):
        strokes, point_bpe = _trace_text(atomic, atomic_to_bpe, delta)
    else:
        raise ValueError(f"Unsupported tokeniser type: {type(discrete)}")

    # Validate against reference decode (without downsample/smooth)
    tokens = tokeniser.convert_ids_to_tokens(ink_token_ids)
    if tokeniser._trained_tokeniser is not None:
        tokens = tokeniser._trained_tokeniser.split(tokens)
    ref_ink_raw = discrete.detokenise(tokens)
    ref_ink = ref_ink_raw.scale(delta)

    _validate_trace(strokes, point_bpe, ref_ink, len(ink_token_ids), repr_id.type.value)

    # Smooth strokes for nicer rendering (preserves point count)
    smoothed: list[np.ndarray] = []
    for s in strokes:
        if len(s) >= 7:  # savgol needs window_length <= len
            sx = np.asarray(savgol_filter(s[:, 0], window_length=7, polyorder=3))
            sy = np.asarray(savgol_filter(s[:, 1], window_length=7, polyorder=3))
            smoothed.append(np.column_stack((sx, sy)))
        else:
            smoothed.append(s)

    return smoothed, point_bpe


def _teacher_forced_decode(
    model: HTRModel,
    instance: Instance,
    capture: AttentionCapture,
) -> str:
    """Run teacher-forced decoding: feed ground truth tokens one at a time.

    Attention is captured at each decode step via the patched MHA layers.
    Returns the ground truth text used for decoding.
    """
    device = next(model.parameters()).device

    # Prefill: embed ink context and build KV cache
    context_emb = model._repr_embedder.embed(instance.repr)  # (seq_len, d_model)
    context_emb = context_emb.unsqueeze(0)  # (1, seq_len, d_model)
    context_len = context_emb.shape[1]

    result = model._forward(
        context_emb,
        start_pos=0,
        kv_caches=None,
        attn_mask=None,
        use_cache=True,
    )
    assert isinstance(result, tuple)
    _, kv_caches = result

    # Ground truth char IDs: [BOS, c1, c2, ..., cN, EOS]
    gt_ids = instance.char.tolist()
    text = instance.parsed.text

    # Teacher-forced decode: feed each GT token, capture attention
    # At step 0: feed BOS, model predicts first char
    # At step 1: feed gt_ids[1] (first GT char), model predicts second char
    # ...
    # At step N: feed gt_ids[N] (last GT char), model predicts EOS
    # We want N+1 steps total (one per output character + EOS)
    for step in range(len(gt_ids) - 1):
        token_id = gt_ids[step]
        token_tensor = torch.tensor([[token_id]], device=device)
        token_emb = model._char_embedder.embed(token_tensor)  # (1, 1, d_model)

        step_pos = torch.tensor([[context_len + step]], device=device)
        result = model._forward(
            token_emb,
            start_pos=step_pos,
            kv_caches=kv_caches,
            attn_mask=None,
            use_cache=True,
        )
        assert isinstance(result, tuple)
        _, kv_caches = result

    capture.flush()
    return text


def run_model(
    repr_id: TokeniserId,
    task: Task = Task.HTR,
) -> tuple[list[np.ndarray], list[list[list[int]]], list[np.ndarray], int, str]:
    """Load model, run teacher-forced inference, return ink + mapping + attention."""
    model_id = ModelId(task=task, repr_id=repr_id)
    model = ModelFactory.load_pretrained(model_id)
    model.eval()
    assert isinstance(model, HTRModel)

    num_layers = len(model._decoder.layers)
    capture = AttentionCapture(num_layers)
    patch_model(model, capture)

    parsed = Parsed.from_path(SAMPLE_PATH)
    instance = Instance.from_parsed(parsed, repr_id=repr_id).to_device()

    ink_len = instance.repr.shape[0]

    with torch.no_grad():
        text = _teacher_forced_decode(model, instance, capture)

    print(f"  Text:           '{text}'")
    print(f"  Ink tokens:     {ink_len}")
    print(f"  Decode steps:   {len(capture.step_weights)}")

    # Build ink and mapping
    ink_ids = instance.repr.tolist()
    strokes, point_bpe = build_ink_and_mapping(repr_id, ink_ids)

    # Compute per-step attention distributions
    distributions = rollout_attention(capture.step_weights)

    # Print percentage of attention on ink vs text tokens
    ink_pcts, bos_pcts, eos_pcts = [], [], []
    for d in distributions:
        total = d.sum()
        ink_pcts.append(d[:ink_len].sum() / total * 100)
        bos_pcts.append(d[0] / total * 100)
        eos_pcts.append(d[ink_len - 1] / total * 100)
    mean_ink = np.mean(ink_pcts)
    mean_bos = np.mean(bos_pcts)
    mean_eos = np.mean(eos_pcts)
    n = len(distributions)
    print(f"  Attention split (avg over {n} steps):")
    print(f"    ink={mean_ink:.1f}%  text={100 - mean_ink:.1f}%")
    print(f"    ink BOS={mean_bos:.1f}%  ink EOS={mean_eos:.1f}%")

    return strokes, point_bpe, distributions, ink_len, text


def _is_dot_stroke(stroke: np.ndarray, atol: float = 1e-6) -> bool:
    """Return True when a stroke is visually a dot.

    Includes true single-point strokes and degenerate strokes with repeated
    identical coordinates (common for TextTokens zero-delta runs).
    """
    if len(stroke) == 0:
        return False
    if len(stroke) == 1:
        return True
    deltas = np.linalg.norm(stroke - stroke[0], axis=1)
    return bool(np.max(deltas) <= atol)


def _compute_segment_attention(
    strokes: list[np.ndarray],
    point_bpe: list[list[list[int]]],
    attn: np.ndarray,
    ink_len: int,
) -> list[np.ndarray]:
    """For each stroke, compute attention value per line segment.

    A segment connects point[i] to point[i+1]. Its attention is the mean
    attention of the BPE tokens that produced point[i+1].
    """
    seg_attns: list[np.ndarray] = []
    for s_idx, stroke in enumerate(strokes):
        if _is_dot_stroke(stroke):
            seg_attns.append(np.array([]))
            continue
        n_segments = len(stroke) - 1
        if n_segments <= 0:
            seg_attns.append(np.array([]))
            continue
        vals = np.zeros(n_segments)
        for seg_i in range(n_segments):
            point_i = seg_i + 1  # endpoint of segment
            bpe_indices = point_bpe[s_idx][point_i]
            if bpe_indices:
                valid = [b for b in bpe_indices if b < ink_len]
                if valid:
                    vals[seg_i] = np.mean([attn[b] for b in valid])
        seg_attns.append(vals)
    return seg_attns


def _get_single_point_strokes(
    strokes: list[np.ndarray],
    point_bpe: list[list[list[int]]],
) -> tuple[np.ndarray, list[list[int]]]:
    """Collect coordinates and BPE indices for dot-like strokes.

    Returns:
        coords: (N, 2) array of dot positions.
        dot_bpes: list of BPE index lists, one per dot.
    """
    coords: list[np.ndarray] = []
    dot_bpes: list[list[int]] = []
    for s_idx, stroke in enumerate(strokes):
        if len(stroke) == 0:
            continue
        if _is_dot_stroke(stroke):
            coords.append(stroke[0])
            # Merge all point-level BPE indices for degenerate repeated-point dots.
            merged = sorted({b for bpes in point_bpe[s_idx] for b in bpes})
            dot_bpes.append(merged)
    if not coords:
        return np.empty((0, 2)), []
    return np.stack(coords), dot_bpes


def _compute_dot_attention(
    dot_bpes: list[list[int]],
    attn: np.ndarray,
    ink_len: int,
) -> np.ndarray:
    """Compute attention value for each single-point stroke."""
    vals = np.zeros(len(dot_bpes))
    for i, bpes in enumerate(dot_bpes):
        valid = [b for b in bpes if b < ink_len]
        if valid:
            vals[i] = np.mean([attn[b] for b in valid])
    return vals


def _build_segments_and_ranges(
    strokes: list[np.ndarray],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build flattened segment array and per-stroke index ranges."""
    all_segments: list[np.ndarray] = []
    stroke_seg_ranges: list[tuple[int, int]] = []
    seg_offset = 0
    for stroke in strokes:
        if _is_dot_stroke(stroke):
            stroke_seg_ranges.append((seg_offset, seg_offset))
            continue
        n_seg = max(len(stroke) - 1, 0)
        if n_seg > 0:
            segments = np.stack([stroke[:-1], stroke[1:]], axis=1)
            all_segments.append(segments)
        stroke_seg_ranges.append((seg_offset, seg_offset + n_seg))
        seg_offset += n_seg
    if not all_segments:
        return np.empty((0, 2, 2)), stroke_seg_ranges
    all_segs_arr = np.concatenate(all_segments, axis=0)
    return all_segs_arr, stroke_seg_ranges


def _draw_completion_prefix(
    ax: Axes,
    completion: str,
    text_len: int,
) -> None:
    """Draw completion prefix in plain black text."""
    display = BOS_TOKEN + completion.replace(" ", SPACE_TOKEN)
    max_chars = len(BOS_TOKEN) + max(text_len, 1)
    ax.set_xlim(0, max_chars)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0,
        0.5,
        display,
        ha="left",
        va="center",
        fontsize=10,
        fontfamily="monospace",
        color="black",
    )


def save_table(
    strokes: list[np.ndarray],
    point_bpe: list[list[list[int]]],
    distributions: list[np.ndarray],
    ink_len: int,
    text: str,
    output_path: str,
) -> None:
    """Create a static PDF table with completion, target char, and ink attention."""
    n_attn_steps = len(distributions)
    n_rows = max(n_attn_steps, len(text) + 1)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
        }
    )

    # Ink bounds (shared across all rows)
    all_pts = np.concatenate([s for s in strokes if len(s) > 0])
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    x_pad = (x_max - x_min) * 0.05 + 1
    y_pad = (y_max - y_min) * 0.05 + 1

    # Precompute segments and single-point strokes
    all_segs_arr, stroke_seg_ranges = _build_segments_and_ranges(strokes)
    n_total_segments = len(all_segs_arr)
    dot_coords, dot_bpes = _get_single_point_strokes(strokes, point_bpe)
    has_dots = len(dot_coords) > 0

    # Colormap and scale
    cmap = plt.get_cmap("Reds")

    # Normalize each step's attention to [0, 1]
    normalized_distributions: list[np.ndarray] = []
    for d in distributions:
        ink_attn = d[:ink_len]
        d_min = ink_attn.min()
        d_max = ink_attn.max()
        if d_max - d_min > 1e-12:
            normed = (d - d_min) / (d_max - d_min)
        else:
            normed = np.zeros_like(d)
        normalized_distributions.append(normed)
    distributions = normalized_distributions
    clim_max = 1.0

    # Layout: derive column widths from content
    row_height = 0.3
    ink_aspect = (x_max - x_min + 2 * x_pad) / max(y_max - y_min + 2 * y_pad, 1e-6)
    ink_width = row_height * ink_aspect
    ink_width = max(3.0, min(8.0, ink_width))  # clamp to reasonable range
    target_char_width = 0.5
    completion_chars = len(BOS_TOKEN) + max(len(text), 1)
    completion_width = max(2.0, min(6.0, 0.12 * completion_chars))
    fig_width = completion_width + target_char_width + ink_width
    fig_height = row_height * n_rows

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        n_rows,
        3,
        width_ratios=[completion_width, target_char_width, ink_width],
        hspace=0.01,
        wspace=0,
    )
    axes = np.empty((n_rows, 3), dtype=object)
    for r in range(n_rows):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c])

    for row in range(n_rows):
        ax_completion = axes[row, 0]
        ax_char = axes[row, 1]
        ax_ink = axes[row, 2]

        if row == 0:
            ax_completion.set_title("Prefix", fontsize=10, pad=6)
            ax_char.set_title("Target", fontsize=10, pad=6)
            ax_ink.set_title("Ink Attention", fontsize=10, pad=6)

        # Completion prefix
        completion = text[: min(row, len(text))]
        _draw_completion_prefix(
            ax_completion,
            completion,
            len(text),
        )

        # Character label
        if row < len(text):
            char = SPACE_TOKEN if text[row] == " " else text[row]
        else:
            char = EOS_TOKEN
        ax_char.text(
            0.5,
            0.5,
            char,
            ha="center",
            va="center",
            fontsize=10,
            fontfamily="monospace",
            transform=ax_char.transAxes,
        )
        ax_char.axis("off")

        # Ink plot
        ax_ink.set_xlim(x_min - x_pad, x_max + x_pad)
        ax_ink.set_ylim(y_min - y_pad, y_max + y_pad)
        ax_ink.invert_yaxis()
        ax_ink.set_aspect("equal", adjustable="box")
        ax_ink.axis("off")

        # Compute attention colors for this step
        if n_attn_steps == 0:
            attn = np.array([])
        else:
            attn = distributions[min(row, n_attn_steps - 1)]
        seg_attns = _compute_segment_attention(strokes, point_bpe, attn, ink_len)
        if n_total_segments > 0:
            flat = np.zeros(n_total_segments)
            for s_idx, sa in enumerate(seg_attns):
                start, end = stroke_seg_ranges[s_idx]
                if len(sa) > 0:
                    flat[start:end] = sa

            lc = LineCollection(
                all_segs_arr.tolist(),
                cmap=cmap,
                linewidths=1.0,
                capstyle="round",
                joinstyle="round",
            )
            lc.set_array(flat)
            lc.set_clim(0, clim_max)
            ax_ink.add_collection(lc)

        # Render dot-like strokes
        if has_dots:
            dot_attns = _compute_dot_attention(dot_bpes, attn, ink_len)
            ax_ink.scatter(
                dot_coords[:, 0],
                dot_coords[:, 1],
                c=dot_attns,
                cmap=cmap,
                vmin=0,
                vmax=clim_max,
                s=4,
                zorder=5,
                edgecolors="none",
            )

    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {output_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        ("HTR ScribeTokens", TokeniserId.create_scribe(), Task.HTR, OUTPUT_HTR_SCRIBE),
        ("HTR TextTokens", TokeniserId.create_text(), Task.HTR, OUTPUT_HTR_TEXT),
        ("HTR_SFT ScribeTokens", TokeniserId.create_scribe(), Task.HTR_SFT, OUTPUT_HTR_SFT_SCRIBE),
    ]

    for label, repr_id, task, output_path in configs:
        print(f"=== {label} ===")
        strokes, bpe, dists, ink_len, text = run_model(repr_id, task)
        save_table(strokes, bpe, dists, ink_len, text, str(output_path))
        print()


if __name__ == "__main__":
    main()
