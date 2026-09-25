# Reproducing the paper figures

Run the commands below from the repository root. The overview is an
illustration, not an additional experiment; the other three figures read the
released numerical results.

## Overview schematic

`figures/overview.pdf` is the paper's blue, open-layout A/B/C schematic.
Panel A illustrates variation in residual family DIF; panel B contrasts
matched-random and low-DIF item sets; panel C illustrates a near-tied ordering
change. Model letters and scores are illustrative. The three model-family
colors are consistent across the two rankings.

Install the plotting dependencies, then regenerate it with:

```bash
python -m pip install -r requirements-figures.txt
python tools/generate_overview.py --output-dir ../audit-data/outputs/overview
```

This writes `overview.pdf`, `overview.svg`, `overview.png`, and
`overview-layout-check.json` outside the released files. The requirements use
the spectral constraints so a newer plotting dependency cannot silently
upgrade NumPy and invalidate strict-tie comparisons. Always check the environment
after installation with `python -m pip check`.

An alternative local output directory is:

```bash
python tools/generate_overview.py --output-dir outputs/overview --stem overview --dpi 300
```

The generator retains the established drawing without redesign: dot counts,
random seed, model ordering, scores, family assignments, colors, line widths,
label weights, and geometry are fixed in the script. Its built-in layout
checks test header alignment, clipping, annotation collisions, and consistent
model colors. They run before any figure is saved.

The reference export was generated with Python 3.12 and Matplotlib 3.11.2,
using Times New Roman. The font preference list falls back to Times,
TeX Gyre Termes, then STIXGeneral when that font is unavailable. A fallback can
change glyph shapes and text metrics; it does not change the illustrative
values. Fonts are embedded in the released PDF and outlined in the SVG, so
using those files does not require the rendering font to be installed.
No font files are redistributed.

For the release port, all drawing and layout functions were preserved.
Against the established reference, the regenerated PNG had identical decoded
pixels (5400 by 1638), the PDF had an identical drawing stream and page box,
and the layout-check report was unchanged. Metadata and output naming were
made anonymous and portable, so file checksums need not match older exports.
PDF author, creator, and producer are `Anonymous`; PDF dates are omitted.
The SVG and PNG also use anonymous creator/software metadata.

Include the vector PDF in LaTeX with:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/overview.pdf}
  \caption{Illustration of benchmark recomposition: low-DIF and matched-random
  subtests match item-group and easiness composition, yet can yield different
  orderings for near-tied models. Colors denote model families in panel C;
  scores are illustrative.}
  \label{fig:overview}
\end{figure}
```

## Empirical result figures

```bash
python -m pip install -c requirements-pinned.txt -e '.[plots]'
python tools/render_paper_figures.py --results-dir results --output-dir outputs/paper_figures
```

The renderer first validates the numerical claims against the result tables.
It then produces:

| File | Numerical input |
| --- | --- |
| `gap_excess_verified.pdf` | `results/gap_sensitivity/gap_sensitivity_summary.csv` |
| `population_robustness_verified.pdf` | `results/population_robustness/robustness_summary.csv` |
| `content_audit_verified.pdf` | `results/content_audit/paired_binary_enrichment.csv` |

It also writes `verification.json`. The full validation uses additional
primary, item-stability, and content-audit files in `results`; do not pass only
the three CSV files listed above. The shipped reference PDFs are in
`figures/`. Regenerating into `outputs/paper_figures` leaves them untouched.

Author and creator metadata are explicitly anonymous. The plotting routines
use fixed layouts and ReportLab's deterministic export mode. Small binary
differences across library versions do not by themselves imply a numerical
or visual change; compare the source tables and rendered pages.
