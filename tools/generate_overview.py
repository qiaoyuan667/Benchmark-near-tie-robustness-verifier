"""Generate the paper's illustrative benchmark-recomposition overview.

The frozen three-panel drawing preserves its model orderings, score values,
family assignments, item counts, colors, typography, and geometry.
Run from the repository root: python tools/generate_overview.py
Dependency: matplotlib==3.11.2.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_hex
from matplotlib.patches import Ellipse, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox


OUT_DIR = Path("figures")
STEM = "overview"
SEED = 27
FIGSIZE = (18.0, 6.0)

WHITE = "#FFFFFF"
INK = "#153449"
SECONDARY = "#56717F"
RULE = "#9BB0BB"
PALE_RULE = "#D6E0E5"
STRIP = "#F2F7FA"
# Clear blue progression for residual DIF, paired with saturated jewel-tone
# family colors. Ordering and mixture proportions are unchanged.
LOW_DIF = "#A9DAF5"
MID_DIF = "#3895DC"
HIGH_DIF = "#2055B0"
HIGHLIGHT = "#FDF0E7"
REVERSAL = "#D7735C"
TRACK = "#EAF1F5"
FAMILY = {
    "blue": "#2F64D6", "coral": "#E65F38", "sage": "#009D87",
}

# Scores/orderings are unchanged. Illustrative family follows model identity,
# not rank; both leaderboards use the same three-color mapping.
MODELS_MR = list("ABCDEFGHIJ")
SCORES_MR = [75.4, 70.4, 69.8, 66.4, 63.1, 60.6, 59.8, 56.8, 53.7, 50.4]
MODELS_LD = ["A", "C", "B", "D", "E", "F", "G", "H", "I", "J"]
SCORES_LD = [75.3, 70.2, 69.7, 66.3, 63.0, 60.5, 59.7, 56.7, 53.6, 50.3]
MODEL_FAMILY = {
    "A": "blue", "B": "coral", "C": "sage", "D": "coral", "E": "sage",
    "F": "blue", "G": "blue", "H": "coral", "I": "sage", "J": "coral",
}
COLORS_MR = [MODEL_FAMILY[model] for model in MODELS_MR]
COLORS_LD = [MODEL_FAMILY[model] for model in MODELS_LD]
BAR_MAX = 80.0
HEADER_BASELINE = .796
HEADER_FONTSIZE = 13.8
PANEL_LABEL_FONTSIZE = 18.0
NEAR_TIE_FONTSIZE = 14.0
RANK_REVERSAL_FONTSIZE = 14.5
MATCHING_FONTSIZE = 18.0
ARROW_WEIGHT_SCALE = 1.45
HORIZONTAL_RULE_WEIGHT_SCALE = 1.45
VERTICAL_PADDING_PT = 2.0

# One shared coordinate per column guarantees header/body center alignment.
MR = {"family": .593, "model": .629, "bar0": .649, "bar1": .716, "score": .758,
      "near_arrow": .721, "near_label": .725}
LD = {"family": .817, "model": .852, "bar0": .872, "bar1": .939, "score": .980,
      "near_arrow": .944, "near_label": .948}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "STIXGeneral"],
    "mathtext.fontset": "stix", "axes.unicode_minus": False,
    "svg.fonttype": "path", "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.hashsalt": "near-tied-refined", "savefig.facecolor": WHITE,
})


def mix(color, target, amount):
    return to_hex(tuple((1 - amount) * a + amount * b
                        for a, b in zip(to_rgb(color), to_rgb(target))))


def text(ax, x, y, s, size, *, weight="normal", color=INK, ha="left",
         va="center", style="normal", gid=None):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                   ha=ha, va=va, fontstyle=style, linespacing=1.02,
                   transform=ax.transAxes, gid=gid, zorder=8)


def line(ax, x1, y1, x2, y2, *, color=RULE, lw=1.0, zorder=2, gid=None):
    if y1 == y2:
        lw *= HORIZONTAL_RULE_WEIGHT_SCALE
    elif x1 != x2:
        # The diagonal segments are part of the two branching arrows.
        lw *= ARROW_WEIGHT_SCALE
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw,
            transform=ax.transAxes, clip_on=False, solid_capstyle="butt", zorder=zorder,
            gid=gid)


def soft_dot(ax, x, y, diameter_pt, color, *, gid=None):
    """Subtle vector shading: circular at any figure aspect ratio."""
    width = diameter_pt / (ax.figure.get_figwidth() * ax.get_position().width * 72)
    height = diameter_pt / (ax.figure.get_figheight() * ax.get_position().height * 72)
    base = Ellipse((x, y), width, height, transform=ax.transAxes,
                   facecolor=mix(color, INK, .04), edgecolor="none", zorder=4, gid=gid)
    ax.add_patch(base)
    for k in range(12):
        f = 1 - k / 14
        shift = (1 - f) * .08
        ax.add_patch(Ellipse((x - width * shift, y + height * shift),
                            width * f, height * f, transform=ax.transAxes,
                            facecolor=mix(color, WHITE, .015 + k * .008),
                            edgecolor="none", zorder=4))
    return base


def dot_palette(n, proportions, rng):
    # Same count rounding and padding policy as the original script.
    palette = []
    for color, share in proportions:
        palette.extend([color] * round(n * share))
    while len(palette) < n:
        palette.append(proportions[-1][0])
    palette = palette[:n]
    rng.shuffle(palette)
    return palette


def dot_cloud(ax, box, n, proportions, *, seed, diameter=16.8, organic=False, rows=3):
    """Deterministic organic cloud or softly jittered three-row strip."""
    rng = random.Random(seed)
    x0, y0, w, h = box
    pts = []
    if organic:
        sx = ax.figure.get_figwidth() * ax.get_position().width * 72
        sy = ax.figure.get_figheight() * ax.get_position().height * 72
        for _ in range(n):
            candidates = []
            for _ in range(100):
                r, angle = math.sqrt(rng.random()), rng.uniform(0, math.tau)
                p = (x0 + w / 2 + w / 2 * r * math.cos(angle),
                     y0 + h / 2 + h / 2 * r * math.sin(angle))
                distance = min((((p[0] - q[0]) * sx) ** 2 + ((p[1] - q[1]) * sy) ** 2
                                for q in pts), default=1)
                candidates.append((distance, p))
            pts.append(max(candidates, key=lambda item: item[0])[1])
    else:
        cols = math.ceil(n / rows)
        for r in range(rows):
            for c in range(cols):
                if len(pts) >= n:
                    break
                pts.append((x0 + (c + .5 + rng.uniform(-.16, .16)) * w / cols,
                            y0 + h - (r + .5 + rng.uniform(-.18, .18)) * h / rows))
    for (x, y), color in zip(pts, dot_palette(n, proportions, rng)):
        soft_dot(ax, x, y, diameter, color)


def arrow(ax, start, end, *, color=RULE, lw=1.35, head=12, style="->", gid=None):
    patch = FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle=style,
                            mutation_scale=head * 1.08, linewidth=lw * ARROW_WEIGHT_SCALE,
                            color=color, gid=gid,
                            connectionstyle="arc3,rad=0", shrinkA=0, shrinkB=0,
                            zorder=6, capstyle="round", joinstyle="round")
    ax.add_patch(patch)
    return patch


def stage_header(ax, letter, x0, x1):
    text(ax, x0, .970, letter, PANEL_LABEL_FONTSIZE, weight="bold", gid=f"panel-label-{letter}")
    line(ax, x0, .933, x1, .933, color=RULE, lw=1.05)


def pill_path(ax, x, y, w, h):
    aspect = (ax.figure.get_figwidth() * ax.get_position().width /
              (ax.figure.get_figheight() * ax.get_position().height))
    ry, rx = h / 2, h / 2 / aspect
    k = .55228475
    v = [(x + rx, y), (x + w - rx, y),
         (x + w - rx + k * rx, y), (x + w, y + ry - k * ry), (x + w, y + ry),
         (x + w, y + ry + k * ry), (x + w - rx + k * rx, y + h), (x + w - rx, y + h),
         (x + rx, y + h), (x + rx - k * rx, y + h), (x, y + ry + k * ry), (x, y + ry),
         (x, y + ry - k * ry), (x + rx - k * rx, y), (x + rx, y), (x + rx, y)]
    codes = [MplPath.MOVETO, MplPath.LINETO] + [MplPath.CURVE4] * 6 + [MplPath.LINETO] + [MplPath.CURVE4] * 6 + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(v, codes), transform=ax.transAxes,
                     facecolor="none", edgecolor="none")


def gradient_bar(ax, x, y, w, h):
    """All-vector gradient with circular ends (no embedded bitmap)."""
    clip = pill_path(ax, x, y, w, h)
    ax.add_patch(clip)
    cmap = LinearSegmentedColormap.from_list("dif", [LOW_DIF, MID_DIF, HIGH_DIF])
    for i in range(256):
        stripe = Rectangle((x + w * i / 256, y), w / 256 + .00001, h,
                           transform=ax.transAxes, facecolor=cmap(i / 255),
                           edgecolor="none", linewidth=0, zorder=1)
        stripe.set_clip_path(clip)
        ax.add_patch(stripe)


def draw_lock(ax, x, y):
    w, h = .016, .043
    color = "#6C91A4"
    # Draw the rounded shackle in physical proportions.
    theta = [math.pi * i / 40 for i in range(41)]
    ax.plot([x + .0055 * math.cos(t) for t in theta],
            [y + h / 2 + .005 + .0165 * math.sin(t) for t in theta],
            color=color, lw=3.0, transform=ax.transAxes, zorder=4)
    line(ax, x - .0055, y + h / 2 + .005, x - .0055, y + h / 2 - .006, color=color, lw=3)
    line(ax, x + .0055, y + h / 2 + .005, x + .0055, y + h / 2 - .006, color=color, lw=3)
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h,
                           transform=ax.transAxes, facecolor=color, edgecolor="none", zorder=5))
    ax.add_patch(Ellipse((x, y + .003), .0032, .0096, transform=ax.transAxes,
                         facecolor=WHITE, edgecolor="none", zorder=7))
    line(ax, x, y + .001, x, y - .010, color=WHITE, lw=1.45, zorder=7)


def vertical_double_arrow(ax, x, label_x, y_top, y_bottom, *, gid):
    arrow(ax, (x, y_bottom), (x, y_top), color=INK, lw=1.15, head=8,
          style="<|-|>", gid=f"{gid}-arrow")
    text(ax, label_x, (y_top + y_bottom) / 2, "near\ntied",
         NEAR_TIE_FONTSIZE, style="italic", weight="bold", gid=gid)


def draw_table(ax):
    x0, x_rank, x_mid, x1 = .536, .571, .793, .997
    top, group_rule, header_rule = .933, .840, .777
    # Expand the row spacing, not the type or symbols, so the bottom rule
    # reaches the lowest content in panels A/B. The header stays fixed.
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    left_floor_px = min(artist.get_window_extent(renderer).y0
                        for artist in (*ax.texts, *ax.lines, *ax.patches)
                        if artist.get_visible())
    bottom = ax.transAxes.inverted().transform((0, left_floor_px))[1]
    first_y = .738
    row_h = (first_y - bottom - .035) / 9
    y2, y3 = first_y - row_h, first_y - 2 * row_h
    line(ax, x0, top, x1, top, color="#517187", lw=1.25)
    line(ax, x_rank, group_rule, x1, group_rule, lw=.90)
    line(ax, x0, header_rule, x1, header_rule, lw=.90)
    line(ax, x0, bottom, x1, bottom, color="#517187", lw=1.25,
         gid="table-bottom-rule")
    line(ax, x_rank, .920, x_rank, bottom, lw=.80)
    # Interrupt the central separator around the crossing arrows and label.
    line(ax, x_mid, .920, x_mid, y2 + .030, lw=.75)
    line(ax, x_mid, y3 - .093, x_mid, bottom, lw=.75)

    ax.add_patch(Rectangle((x0, first_y - 2.5 * row_h), x1 - x0, 2 * row_h,
                           transform=ax.transAxes, facecolor=HIGHLIGHT, edgecolor="none", zorder=0))
    text(ax, (x0 + x_rank) / 2, HEADER_BASELINE, "Rank", HEADER_FONTSIZE,
         weight="bold", ha="center", va="baseline", gid="rank-header")
    text(ax, (x_rank + x_mid) / 2, .889, "Matched Random", 23, weight="bold", ha="center")
    text(ax, (x_mid + x1) / 2, .889, "Low-DIF", 23, weight="bold", ha="center")

    for name, cols in (("mr", MR), ("ld", LD)):
        for key, label in (("family", "Family"), ("model", "Model"), ("score", "Score")):
            text(ax, cols[key], HEADER_BASELINE, label, HEADER_FONTSIZE,
                 weight="bold", ha="center", va="baseline", gid=f"{name}-{key}-header")

    for i in range(10):
        y = first_y - i * row_h
        text(ax, (x0 + x_rank) / 2, y, str(i + 1), 14.5, ha="center")
        for name, cols, model, score, family_key in (
            ("mr", MR, MODELS_MR[i], SCORES_MR[i], COLORS_MR[i]),
            ("ld", LD, MODELS_LD[i], SCORES_LD[i], COLORS_LD[i]),
        ):
            soft_dot(ax, cols["family"], y, 17.2, FAMILY[family_key], gid=f"{name}-family-{i}")
            text(ax, cols["model"], y, model, 15.5, ha="center", gid=f"{name}-model-{i}")
            track_w = cols["bar1"] - cols["bar0"]
            for width, color in ((track_w, TRACK), (track_w * score / BAR_MAX, FAMILY[family_key])):
                ax.add_patch(Rectangle((cols["bar0"], y - .014), width, .028,
                                       transform=ax.transAxes, facecolor=color, edgecolor="none", zorder=3))
            text(ax, cols["score"], y, f"{score:.1f}", 14.5, ha="center", gid=f"{name}-score-{i}")

    for name, cols in (("mr", MR), ("ld", LD)):
        for upper in (1, 5):
            vertical_double_arrow(ax, cols["near_arrow"], cols["near_label"],
                                  first_y - upper * row_h - .006,
                                  first_y - (upper + 1) * row_h + .006,
                                  gid=f"{name}-near-tie-{upper}")
    arrow(ax, (.778, y2), (.806, y3), color=REVERSAL, head=10.5, lw=1.55,
          gid="reversal-down-arrow")
    arrow(ax, (.778, y3), (.806, y2), color=REVERSAL, head=10.5, lw=1.55,
          gid="reversal-up-arrow")
    text(ax, .7895, y3 - .052, "rank\nreversal", RANK_REVERSAL_FONTSIZE, color=REVERSAL,
         style="italic", weight="bold", ha="center", gid="rank-reversal")


def build_figure():
    fig = plt.figure(figsize=FIGSIZE, dpi=160, facecolor=WHITE)
    # Leave a modest right canvas margin, balanced with the left panel inset.
    ax = fig.add_axes([0, 0, .980, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    stage_header(ax, "A", .019, .233)
    stage_header(ax, "B", .255, .508)
    stage_header(ax, "C", .536, .997)

    text(ax, .019, .860, "Benchmark items", 28, weight="bold")
    dot_cloud(ax, (.027, .293, .160, .462), 50,
              [(LOW_DIF, .34), (MID_DIF, .38), (HIGH_DIF, .28)],
              seed=SEED, diameter=17.4, organic=True)
    text(ax, .022, .203, "residual family DIF", 16)
    gradient_bar(ax, .022, .141, .166, .038)
    text(ax, .022, .104, "low", 15)
    text(ax, .188, .104, "high", 15, ha="right")

    line(ax, .194, .577, .222, .650, lw=1.5)
    arrow(ax, (.222, .650), (.246, .650), lw=1.5)
    line(ax, .194, .411, .222, .327, lw=1.5)
    arrow(ax, (.222, .327), (.246, .327), lw=1.5)

    text(ax, .255, .860, "Matched Random", 27, weight="bold")
    text(ax, .255, .508, "Low-DIF", 27, weight="bold")
    for y in (.589, .236):
        ax.add_patch(Rectangle((.255, y), .248, .211, transform=ax.transAxes,
                               facecolor=STRIP, edgecolor="none", zorder=0))
    dot_cloud(ax, (.264, .608, .230, .171), 30,
              [(LOW_DIF, .34), (MID_DIF, .38), (HIGH_DIF, .28)],
              seed=SEED + 1, diameter=16.8)
    dot_cloud(ax, (.264, .255, .230, .171), 30,
              [(LOW_DIF, .78), (MID_DIF, .22)], seed=SEED + 2, diameter=16.8)
    draw_lock(ax, .327, .127)
    text(ax, .346, .129, "item group · easiness", MATCHING_FONTSIZE,
         color=SECONDARY, gid="matching-label")
    arrow(ax, (.506, .650), (.529, .650), lw=1.5)
    arrow(ax, (.506, .327), (.529, .327), lw=1.5)
    draw_table(ax)
    return fig


def validate_layout(fig):
    """Fail if column headers drift, adjacent headings collide, or text clips."""
    assert len(set(FAMILY.values())) == 3
    assert dict(zip(MODELS_MR, COLORS_MR)) == dict(zip(MODELS_LD, COLORS_LD))
    assert MODEL_FAMILY["B"] != MODEL_FAMILY["C"]
    assert MODEL_FAMILY["F"] == MODEL_FAMILY["G"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    tagged = {t.get_gid(): t for t in ax.texts if t.get_gid()}
    headers = [t for gid, t in tagged.items() if gid.endswith("-header")]
    panel_labels = [tagged[f"panel-label-{letter}"] for letter in "ABC"]
    assert all(t.get_fontsize() == PANEL_LABEL_FONTSIZE for t in panel_labels)
    assert all(t.get_position()[1] == .970 and t.get_fontweight() == "bold" for t in panel_labels)
    assert len(headers) == 7
    assert all(t.get_va() == "baseline" for t in headers)
    assert all(t.get_position()[1] == HEADER_BASELINE for t in headers)
    assert all(t.get_fontsize() == HEADER_FONTSIZE for t in headers)
    max_error = 0.0
    for name in ("mr", "ld"):
        for key in ("model", "score"):
            header = tagged[f"{name}-{key}-header"].get_window_extent(renderer)
            hc = (header.x0 + header.x1) / 2
            for i in range(10):
                item = tagged[f"{name}-{key}-{i}"].get_window_extent(renderer)
                error = abs((item.x0 + item.x1) / 2 - hc)
                assert error < .01, (name, key, i, error)
                max_error = max(max_error, error)
        family_header = tagged[f"{name}-family-header"].get_window_extent(renderer)
        model_header = tagged[f"{name}-model-header"].get_window_extent(renderer)
        assert family_header.x1 + 3 < model_header.x0, "Family/Model headers overlap"
        coords = MR if name == "mr" else LD
        family_center = ax.transAxes.transform((coords["family"], 0))[0]
        assert abs((family_header.x0 + family_header.x1) / 2 - family_center) < .01
    for t in ax.texts:
        bb = t.get_window_extent(renderer)
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= fig.bbox.x1 and bb.y1 <= fig.bbox.y1, t.get_text()
    low_labels = [t for t in ax.texts if t.get_text() in ("low", "high")]
    left_floor = min(t.get_window_extent(renderer).y0 for t in low_labels)
    bottom_rule = next(l for l in ax.lines if l.get_gid() == "table-bottom-rule")
    rule_y = bottom_rule.get_window_extent(renderer).y0
    bottom_alignment_error = abs(rule_y - left_floor)
    assert bottom_alignment_error < .01, "Table bottom is not aligned with the left legend"
    assert all(t.get_window_extent(renderer).y0 >= rule_y - .01 for t in ax.texts)
    # Bold annotations have dedicated space: check their full text bounds
    # against every other label and all arrows/rules, not just glyph centers.
    annotation_labels = [t for t in ax.texts
                         if t.get_text() in ("near\ntied", "rank\nreversal")]
    clearance_px = fig.dpi / 72
    for label in annotation_labels:
        assert label.get_fontweight() == "bold"
        bb = label.get_window_extent(renderer).padded(clearance_px)
        for other in ax.texts:
            if other is not label:
                assert not bb.overlaps(other.get_window_extent(renderer)), (
                    "Annotation/text collision", label.get_gid(), other.get_text())
        obstacles = [p for p in ax.patches
                     if isinstance(p, FancyArrowPatch) or "-family-" in (p.get_gid() or "")]
        for other in (*ax.lines, *obstacles):
            stroke_px = other.get_linewidth() * fig.dpi / 144
            assert not bb.overlaps(other.get_window_extent(renderer).padded(stroke_px)), (
                "Annotation/graphic collision", label.get_gid(), other.get_gid())
    near_score_gaps = []
    for name in ("mr", "ld"):
        for upper in (1, 5):
            label_bb = tagged[f"{name}-near-tie-{upper}"].get_window_extent(renderer)
            near_score_gaps.extend(
                (tagged[f"{name}-score-{i}"].get_window_extent(renderer).x0 - label_bb.x1)
                * 72 / fig.dpi for i in (upper, upper + 1))
    return {"model_score_header_max_center_error_px": max_error,
            "table_bottom_matches_left_content_error_px": bottom_alignment_error,
            "table_bottom_is_lowest_rule": True,
            "table_bottom_axes_y": float(bottom_rule.get_ydata()[0]),
            "all_seven_headers_share_baseline": True,
            "header_baseline": HEADER_BASELINE, "header_fontsize_pt": HEADER_FONTSIZE,
            "panel_label_fontsize_pt": PANEL_LABEL_FONTSIZE,
            "near_tie_fontsize_pt": NEAR_TIE_FONTSIZE,
            "rank_reversal_fontsize_pt": RANK_REVERSAL_FONTSIZE,
            "matching_label_fontsize_pt": MATCHING_FONTSIZE,
            "near_tie_and_reversal_bold": True,
            "annotation_collisions": False,
            "annotation_min_clearance_pt": 1.0,
            "near_tie_score_min_horizontal_gap_pt": min(near_score_gaps),
            "arrow_weight_scale": ARROW_WEIGHT_SCALE,
            "horizontal_rule_weight_scale": HORIZONTAL_RULE_WEIGHT_SCALE,
            "family_headers_aligned": True, "header_collisions": False,
            "text_clipped": False, "models_mr": MODELS_MR, "models_ld": MODELS_LD,
            "scores_mr": SCORES_MR, "scores_ld": SCORES_LD,
            "bar_max": BAR_MAX, "dot_counts": [50, 30, 30],
            "family_palette": FAMILY, "model_family": MODEL_FAMILY,
            "same_model_color_in_both_lists": True}


def vertical_export_bounds(fig):
    """Trim only the top/bottom whitespace; keep the full original width."""
    renderer = fig.canvas.get_renderer()
    extents = [artist.get_window_extent(renderer)
               for ax in fig.axes
               for artist in (*ax.texts, *ax.lines, *ax.patches)
               if artist.get_visible()]
    content = Bbox.union(extents).transformed(fig.dpi_scale_trans.inverted())
    padding = VERTICAL_PADDING_PT / 72
    return Bbox.from_extents(0, max(0, content.y0 - padding),
                             fig.get_figwidth(),
                             min(fig.get_figheight(), content.y1 + padding))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--stem", default=STEM, help="Output filename stem without a directory.")
    args = parser.parse_args()
    if args.dpi < 1 or not args.stem or Path(args.stem).name != args.stem or args.stem in {".", ".."}:
        parser.error("dpi must be positive and stem must be a plain filename")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    qa = validate_layout(fig)
    bounds = vertical_export_bounds(fig)
    qa["export_size_inches"] = [bounds.width, bounds.height]
    qa["vertical_padding_pt"] = VERTICAL_PADDING_PT
    qa["horizontal_margins_unchanged"] = True
    for ext in ("pdf", "png", "svg"):
        path = args.output_dir / f"{args.stem}.{ext}"
        kwargs = {"bbox_inches": bounds, "pad_inches": 0, "facecolor": WHITE}
        if ext == "png":
            kwargs["dpi"] = args.dpi
            kwargs["metadata"] = {"Software": "Anonymous"}
        if ext == "pdf":
            kwargs["metadata"] = {"Title": "Benchmark recomposition overview", "Author": "Anonymous",
                                  "Creator": "Anonymous", "Producer": "Anonymous",
                                  "CreationDate": None, "ModDate": None}
        if ext == "svg":
            kwargs["metadata"] = {"Date": None, "Creator": "Anonymous", "Title": "Benchmark recomposition overview"}
        fig.savefig(path, **kwargs)
        print(path)
    (args.output_dir / f"{args.stem}-layout-check.json").write_text(json.dumps(qa, indent=2) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
