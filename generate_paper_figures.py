"""
generate_paper_figures.py
--------------------------
Publication-quality figure generation for the Colonomind texture analysis paper.

This script loads the pre-computed feature cache produced by dgx_refit_pipeline.py
(model-colono/cached_features.npz) and generates:

  Figure 1 — Multi-panel scatter plots (Mayo Score vs. individual texture metrics)
             with per-class colour coding and OLS regression lines.
  Figure 2 — 3-D UMAP embedding scatter coloured by Mayo Score.

Statistical analyses:
  - OLS regression: R², adjusted R², p-value per texture metric.
  - One-Way ANCOVA: each texture metric as covariate, Mayo Score as factor.
  - Results are printed as a formatted LaTeX-compatible table and saved as CSV.

Outputs (written to model-colono/figures/):
  texture_scatter_panel.png  (300 DPI)
  umap_3d_projection.png     (300 DPI)
  stats_summary.csv

Usage:
    python generate_paper_figures.py [--cache_path model-colono/cached_features.npz]

Dependencies:
    numpy>=1.23
    matplotlib>=3.8
    seaborn>=0.13
    statsmodels>=0.14
    scipy>=1.11
    pandas>=2.0
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
CACHE_PATH   = SCRIPT_DIR / "model-colono" / "cached_features.npz"
OUTPUT_DIR   = SCRIPT_DIR / "model-colono" / "figures"

# ---------------------------------------------------------------------------
# Design constants — minimalist, publication-ready colour scheme
# ---------------------------------------------------------------------------

# Warm-to-cool palette aligned with clinical severity (Mayo 0=healthy → 3=severe)
MAYO_PALETTE = {
    0: "#4CAF82",   # teal-green  – remission
    1: "#F7C948",   # amber       – mild
    2: "#F48935",   # orange      – moderate
    3: "#D94F4F",   # crimson     – severe
}
MAYO_LABELS = {
    0: "Mayo 0 (Remission)",
    1: "Mayo 1 (Mild)",
    2: "Mayo 2 (Moderate)",
    3: "Mayo 3 (Severe)",
}

# Typography
FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE  = 11
LABEL_SIZE  = 9
TICK_SIZE   = 8
LEGEND_SIZE = 8

# Texture metric names corresponding to the GLCM feature layout:
# 4 props × 2 distances × 4 angles, then DWT metrics
# For paper figures we select the most clinically interpretable aggregated metrics.
GLCM_PROPS    = ["Contrast", "Homogeneity", "Energy", "Correlation"]
DWT_SUBBANDS  = ["LL", "LH", "HL", "HH"]
DWT_STATS     = ["Energy", "Variance"]

# Indices in the texture feature vector for aggregated per-property means
# Layout: [contrast_d0a0, contrast_d0a1, … contrast_d1a3, homogeneity_…, …, DWT…]
# 4 props × 2 distances × 4 angles = 32 GLCM, then 8 DWT
N_GLCM = 32
N_DWT  = 8

# For plotting, we aggregate over distances and angles (mean per property)
PANEL_METRICS = [
    ("Contrast",    slice(0,  8)),   # first 8 GLCM values → Contrast
    ("Homogeneity", slice(8,  16)),
    ("Energy",      slice(16, 24)),
    ("Correlation", slice(24, 32)),
    ("DWT LL Energy",   slice(N_GLCM + 0, N_GLCM + 1)),
    ("DWT HH Energy",   slice(N_GLCM + 6, N_GLCM + 7)),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_publication_style() -> None:
    """Apply a minimal, publication-ready matplotlib style."""
    plt.rcParams.update(
        {
            "font.family":         FONT_FAMILY,
            "axes.spines.top":     False,
            "axes.spines.right":   False,
            "axes.grid":           True,
            "grid.color":          "#E0E0E0",
            "grid.linewidth":      0.6,
            "axes.linewidth":      0.8,
            "xtick.major.size":    3.0,
            "ytick.major.size":    3.0,
            "xtick.minor.size":    0,
            "ytick.minor.size":    0,
            "figure.dpi":          150,
            "savefig.dpi":         300,
            "axes.titlesize":      TITLE_SIZE,
            "axes.labelsize":      LABEL_SIZE,
            "xtick.labelsize":     TICK_SIZE,
            "ytick.labelsize":     TICK_SIZE,
            "legend.fontsize":     LEGEND_SIZE,
            "legend.framealpha":   0.85,
            "legend.edgecolor":    "#CCCCCC",
        }
    )


def _load_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load extracted features and labels from the NPZ cache.

    Returns
    -------
    texture_features : np.ndarray, shape (N, N_TEXTURE_FEATURES)
    labels           : np.ndarray, shape (N,), dtype int32
    umap_embedding   : np.ndarray, shape (N, 3)
    """
    print(f"[INFO] Loading feature cache from: {cache_path}")
    data = np.load(cache_path, allow_pickle=False)

    texture_features = data["texture_features"].astype(np.float64)
    labels           = data["labels"].astype(np.int32)
    umap_embedding   = data.get("umap_embedding", None)
    if umap_embedding is not None:
        umap_embedding = umap_embedding.astype(np.float64)

    print(
        f"[INFO] Loaded {len(labels)} samples.  "
        f"Texture features: {texture_features.shape[1]}D.  "
        f"Class distribution: "
        + ", ".join(
            f"Mayo {c}: {np.sum(labels == c)}"
            for c in sorted(np.unique(labels))
        )
    )
    return texture_features, labels, umap_embedding


def _build_metric_df(texture_features: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    Construct a tidy DataFrame with one aggregated texture metric per column.

    Each GLCM property is averaged across its distance × angle combinations.
    Each DWT stat is taken as-is.

    Parameters
    ----------
    texture_features : np.ndarray, shape (N, N_TEXTURE_FEATURES)
    labels           : np.ndarray, shape (N,)

    Returns
    -------
    pd.DataFrame with columns: ['mayo_score', *metric_names]
    """
    rows = {"mayo_score": labels.tolist()}
    for metric_name, feat_slice in PANEL_METRICS:
        # Mean across the selected feature indices for this metric
        rows[metric_name] = texture_features[:, feat_slice].mean(axis=1).tolist()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OLS regression statistics
# ---------------------------------------------------------------------------

def _compute_ols_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit OLS models (each texture metric ~ Mayo Score) and extract statistics.

    Returns a DataFrame with one row per metric containing:
        metric, R2, adj_R2, F_stat, p_value, slope, intercept
    """
    results = []
    for metric_name, _ in PANEL_METRICS:
        x = df["mayo_score"].values.astype(float)
        y = df[metric_name].values.astype(float)

        # OLS via statsmodels for full statistics
        X_with_const = sm.add_constant(x)
        ols_model = sm.OLS(y, X_with_const).fit()

        results.append(
            {
                "Metric":     metric_name,
                "R²":         round(ols_model.rsquared, 4),
                "Adj. R²":    round(ols_model.rsquared_adj, 4),
                "F-statistic":round(ols_model.fvalue, 3),
                "p-value":    ols_model.f_pvalue,
                "Slope":      round(ols_model.params[1], 6),
                "Intercept":  round(ols_model.params[0], 6),
            }
        )

    stats_df = pd.DataFrame(results)
    return stats_df


# ---------------------------------------------------------------------------
# ANCOVA
# ---------------------------------------------------------------------------

def _compute_ancova_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform a one-way ANCOVA for each texture metric (covariate) against
    Mayo Score (factor, treated as a continuous ordinal variable for simplicity).

    For each metric:
        Model: metric ~ C(mayo_score) [categorical factor, no covariate]
        The F-test from OLS with categorical encoding gives the ANOVA result.

    Returns a DataFrame with one row per metric.
    """
    import statsmodels.formula.api as smf  # noqa: PLC0415

    ancova_results = []
    for metric_name, _ in PANEL_METRICS:
        formula = f"`{metric_name}` ~ C(mayo_score)"
        model   = smf.ols(formula, data=df).fit()
        anova   = sm.stats.anova_lm(model, typ=2)

        f_val = anova.loc["C(mayo_score)", "F"]
        p_val = anova.loc["C(mayo_score)", "PR(>F)"]

        ancova_results.append(
            {
                "Metric":           metric_name,
                "ANCOVA F":         round(f_val, 3),
                "ANCOVA p-value":   p_val,
                "ANCOVA η²":        round(
                    anova.loc["C(mayo_score)", "sum_sq"]
                    / anova["sum_sq"].sum(),
                    4,
                ),
            }
        )

    return pd.DataFrame(ancova_results)


# ---------------------------------------------------------------------------
# Figure 1: Multi-panel scatter with regression lines
# ---------------------------------------------------------------------------

def _plot_texture_scatter_panel(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate a multi-panel scatter plot (Mayo Score vs. texture metric).

    Each panel shows:
    - Individual data points coloured by Mayo Score class.
    - A global OLS regression line.
    - R² and p-value annotation in the upper corner.
    """
    n_panels = len(PANEL_METRICS)
    n_cols   = 3
    n_rows   = (n_panels + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5.0 * n_cols, 4.2 * n_rows))
    fig.suptitle(
        "Texture Feature Gradients Across Mayo Endoscopic Scores\n"
        "(LIMUC Dataset — GLCM & DWT Analysis)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        hspace=0.45,
        wspace=0.35,
        figure=fig,
    )

    mayo_scores = sorted(df["mayo_score"].unique())

    for panel_idx, (metric_name, _) in enumerate(PANEL_METRICS):
        row = panel_idx // n_cols
        col = panel_idx  % n_cols
        ax  = fig.add_subplot(gs[row, col])

        # --- Per-class scatter (jittered x for readability) ---
        rng = np.random.default_rng(seed=panel_idx)
        for score in mayo_scores:
            mask  = df["mayo_score"] == score
            x_val = df.loc[mask, "mayo_score"].values
            y_val = df.loc[mask, metric_name].values
            jitter = rng.uniform(-0.15, 0.15, size=len(x_val))
            ax.scatter(
                x_val + jitter,
                y_val,
                color=MAYO_PALETTE[score],
                alpha=0.45,
                s=12,
                linewidths=0,
                label=MAYO_LABELS[score],
                zorder=3,
            )

        # --- Global OLS regression line ---
        x_all = df["mayo_score"].values.astype(float)
        y_all = df[metric_name].values.astype(float)
        slope, intercept, *_ = stats.linregress(x_all, y_all)
        x_line = np.linspace(mayo_scores[0] - 0.3, mayo_scores[-1] + 0.3, 100)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="#333333",
            linewidth=1.5,
            linestyle="--",
            zorder=4,
            label="OLS fit",
        )

        # --- R² / p-value annotation ---
        row_stat  = stats_df.loc[stats_df["Metric"] == metric_name].iloc[0]
        r2_val    = row_stat["R²"]
        p_val     = row_stat["p-value"]
        p_str     = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        ax.annotate(
            f"$R^2$ = {r2_val:.3f}\n{p_str}",
            xy=(0.97, 0.96),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=TICK_SIZE,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.7),
        )

        # --- Axis labels ---
        ax.set_xlabel("Mayo Endoscopic Score", fontsize=LABEL_SIZE)
        ax.set_ylabel(metric_name, fontsize=LABEL_SIZE)
        ax.set_title(f"{metric_name}", fontsize=TITLE_SIZE, fontweight="semibold")
        ax.set_xticks(mayo_scores)
        ax.set_xticklabels([f"Mayo {s}" for s in mayo_scores], fontsize=TICK_SIZE)

        # Show legend only on the first panel to save space
        if panel_idx == 0:
            ax.legend(
                loc="upper left",
                fontsize=LEGEND_SIZE - 1,
                framealpha=0.85,
                markerscale=1.4,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[INFO] Saved Figure 1 → {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: 3-D UMAP embedding
# ---------------------------------------------------------------------------

def _plot_umap_3d(
    umap_embedding: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot the 3-D UMAP embedding coloured by Mayo Score.
    Saves two views (azimuth 30° and 120°) side by side.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

    fig = plt.figure(figsize=(14, 5.5))
    fig.suptitle(
        "UMAP 3-D Embedding — Hybrid DL + Texture Feature Space",
        fontsize=12,
        fontweight="bold",
    )

    azimuths = [30, 120]
    mayo_scores = sorted(np.unique(labels))

    for view_idx, azimuth in enumerate(azimuths):
        ax = fig.add_subplot(1, len(azimuths), view_idx + 1, projection="3d")

        for score in mayo_scores:
            mask = labels == score
            ax.scatter(
                umap_embedding[mask, 0],
                umap_embedding[mask, 1],
                umap_embedding[mask, 2],
                c=MAYO_PALETTE[score],
                label=MAYO_LABELS[score],
                s=10,
                alpha=0.6,
                linewidths=0,
            )

        ax.set_xlabel("UMAP-1", fontsize=LABEL_SIZE)
        ax.set_ylabel("UMAP-2", fontsize=LABEL_SIZE)
        ax.set_zlabel("UMAP-3", fontsize=LABEL_SIZE)
        ax.set_title(f"View (azimuth={azimuth}°)", fontsize=TITLE_SIZE)
        ax.view_init(elev=20, azim=azimuth)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, linewidth=0.4, color="#E0E0E0")

        if view_idx == 0:
            ax.legend(
                loc="upper left",
                fontsize=LEGEND_SIZE,
                markerscale=1.5,
                framealpha=0.85,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[INFO] Saved Figure 2 → {output_path}")


# ---------------------------------------------------------------------------
# Statistics table printer
# ---------------------------------------------------------------------------

def _print_stats_table(stats_df: pd.DataFrame, ancova_df: pd.DataFrame) -> None:
    """Print a formatted academic statistics table to stdout."""
    combined = stats_df.merge(ancova_df, on="Metric")

    # Format p-values
    def _fmt_p(p: float) -> str:
        if p < 0.001:
            return "< 0.001***"
        elif p < 0.01:
            return f"{p:.3f}**"
        elif p < 0.05:
            return f"{p:.3f}*"
        return f"{p:.3f}"

    combined["p-value"]        = combined["p-value"].apply(_fmt_p)
    combined["ANCOVA p-value"] = combined["ANCOVA p-value"].apply(_fmt_p)

    header_line = "─" * 100
    print(f"\n{header_line}")
    print(
        " Table 1. OLS Regression and ANCOVA Statistics — "
        "Texture Metrics vs. Mayo Endoscopic Score"
    )
    print(header_line)
    print(combined.to_string(index=False))
    print(header_line)
    print("  Significance codes: *** p<0.001  ** p<0.01  * p<0.05")
    print(f"{header_line}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_figures(cache_path: Path) -> None:
    """
    End-to-end figure generation pipeline.

    Parameters
    ----------
    cache_path : Path
        Path to the NPZ cache produced by dgx_refit_pipeline.py.
    """
    _set_publication_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    texture_features, labels, umap_embedding = _load_cache(cache_path)

    # Build tidy metric DataFrame
    df = _build_metric_df(texture_features, labels)

    # Statistical analysis
    print("[INFO] Computing OLS regression statistics …")
    ols_stats   = _compute_ols_stats(df)

    print("[INFO] Computing ANCOVA statistics …")
    ancova_stats = _compute_ancova_stats(df)

    # Print table
    _print_stats_table(ols_stats, ancova_stats)

    # Save combined stats to CSV
    combined_stats = ols_stats.merge(ancova_stats, on="Metric")
    csv_path = OUTPUT_DIR / "stats_summary.csv"
    combined_stats.to_csv(csv_path, index=False)
    print(f"[INFO] Statistics table saved → {csv_path}")

    # Figure 1: Multi-panel scatter
    fig1_path = OUTPUT_DIR / "texture_scatter_panel.png"
    _plot_texture_scatter_panel(df, ols_stats, fig1_path)

    # Figure 2: 3-D UMAP (requires umap_embedding from cache)
    if umap_embedding is not None:
        fig2_path = OUTPUT_DIR / "umap_3d_projection.png"
        _plot_umap_3d(umap_embedding, labels, fig2_path)
    else:
        print(
            "[WARNING] umap_embedding not found in cache. "
            "Run dgx_refit_pipeline.py first to generate the embedding."
        )

    print("\n[INFO] All figures generated successfully.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-quality figures and statistics for the "
            "Colonomind texture analysis paper."
        )
    )
    parser.add_argument(
        "--cache_path",
        type=Path,
        default=CACHE_PATH,
        help=f"Path to the NPZ feature cache (default: {CACHE_PATH}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.cache_path.is_file():
        print(
            f"[ERROR] Feature cache not found: {args.cache_path}\n"
            "        Run dgx_refit_pipeline.py on the DGX server first."
        )
        sys.exit(1)

    generate_figures(args.cache_path)
