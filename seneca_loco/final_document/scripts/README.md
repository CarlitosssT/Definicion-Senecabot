# `final_document/` — Thesis figure pipeline

Generates every figure in the thesis from raw CSV data. Output is **vectorial PDF** by default (best for LaTeX) and every figure has a unique, traceable filename.

## Layout

```
final_document/
├── data/            # raw CSVs (one row per observation; tidy format preferred)
├── images/          # generated figures (+ _manifest.tsv ledger)
└── scripts/
    ├── style.py             # rcParams, palettes, sizes  ── EDIT for look & feel
    ├── plotters.py          # reusable plot functions    ── ADD new plot types
    ├── io_utils.py          # load_csv, save_figure, naming convention
    ├── registry.py          # list of every figure       ── ADD entries here
    ├── generate_all.py      # CLI: regenerate all/some figures
    └── make_sample_data.py  # writes example CSVs (toy data)
```

## Quick start

```bash
cd final_document
python scripts/make_sample_data.py     # one-time: generate example CSVs
python scripts/generate_all.py         # produces every PDF in images/
```

Filter by id glob, switch format, or wipe images/ first:

```bash
python scripts/generate_all.py "training__*"      # only training figures
python scripts/generate_all.py --fmt png          # raster output
python scripts/generate_all.py --latex            # render text via LaTeX
python scripts/generate_all.py --clean            # delete images/ first
```

## Naming convention (mandatory)

```
<category>__<descriptor>__<variant>.pdf
```

* **Double underscores** separate tokens (single `_` is a math operator in LaTeX).
* Allowed `category` values live in `io_utils.CATEGORIES`.
* `variant` lets you keep multiple takes of the same figure (e.g. `v1`, `seed_avg`, `by_method`).

Examples already in the registry:

| fig_id | source |
|---|---|
| `training__ppo_reward__seed_avg` | `ppo_training_log.csv` |
| `training__ppo_reward__by_method` | `ppo_training_log.csv` |
| `trajectory__hindlimb_joints__sim_vs_mocap` | `mocap_reference.csv`, `sim_rollout.csv` |
| `error__tracking_rmse__per_joint` | `tracking_error.csv` |

`images/_manifest.tsv` logs every saved figure with its source CSVs, plotter function, kwargs hash, and timestamp — so any figure can be traced back to the exact data and code that produced it.

## Using a figure in LaTeX

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{images/training__ppo_reward__by_method.pdf}
  \caption{PPO reward across training, baseline vs.\ imitation.}
  \label{fig:ppo_reward_by_method}
\end{figure}
```

Width fits exactly because `style.COLUMN_WIDTH_IN` is set to your template's `\columnwidth`. To verify, in your `.tex` run `\the\columnwidth`, divide pts by 72.27, and update the constant in `style.py`.

## Adding a new figure (the only workflow you need)

1. Drop the CSV into `data/`.
2. Open `scripts/registry.py`, append a `FigureSpec(...)`. Pick a unique `(category, descriptor, variant)`.
3. If no existing plotter fits, add one to `scripts/plotters.py`.
4. Run `python scripts/generate_all.py "<your_pattern>"`.

## Style flexibility

`style.StyleConfig` exposes every commonly tweaked knob (font family/size, palette, line widths, grid, LaTeX rendering, DPI). For one-off overrides per figure, every plotter accepts a `style_overrides` dict:

```python
plot_training_curves(df, x="step", y="reward",
                     style_overrides={"axes.grid": False, "font.size": 11})
```

Three palettes ship: `wong` (colorblind-safe, default), `muted` (academic), `vivid` (slides). Add yours to `style.PALETTES`.

## Data-amount flexibility

Plotters never hard-code column names or counts. Pass any columns:

* `plot_training_curves` — optional `seed_col` for shaded ±1σ bands, optional `group_col` for multi-line plots.
* `plot_trajectory_comparison` — any number of input DataFrames, any subset of DOF columns, auto-laid-out subplot grid.
* `plot_tracking_error`, `plot_metric_comparison` — accept arbitrary column lists.

Add new plotters following the same contract: take `df` + column-name kwargs, return `(fig, ax)`, accept `style_overrides`.
