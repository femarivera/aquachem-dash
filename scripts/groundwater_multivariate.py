"""
groundwater_multivariate.py
───────────────────────────
Performs hierarchical cluster analysis (HCA) and PCA on groundwater chemistry.
Writes cluster assignments back into the input CSV as a new 'Cluster' column.

Usage:
    python groundwater_multivariate.py  your_data.csv

Settings (edit the CONFIG block below):
    N_CLUSTERS      – number of clusters to cut the dendrogram into
    LINKAGE_METHOD  – ward | complete | average | single
    NAME_COL        – name of the sample-ID column (default: first column)
    NUMERIC_COLS    – list of columns to include; empty = all numeric columns
    LOG_TRANSFORM   – True to log10-transform concentrations before scaling
    LOG_EXCLUDE     – columns to skip when log-transforming (pH, T, etc.)
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — edit these values
# ══════════════════════════════════════════════════════════════════════════════
N_CLUSTERS     = 4
LINKAGE_METHOD = "ward"       # ward | complete | average | single
NAME_COL       = None         # None = use first column automatically
NUMERIC_COLS   = []           # [] = use ALL numeric columns except NAME_COL
LOG_TRANSFORM  = False        # set True to log10-transform before scaling
LOG_EXCLUDE    = ["pH", "T", "Temp", "Temperature"]  # never log these

PALETTE  = ["#185FA5", "#D85A30", "#1D9E75", "#BA7517",
            "#993556", "#534AB7", "#3B6D11", "#A32D2D"]
MARKERS  = ["o", "^", "s", "D", "v", "P", "X", "*"]
# ══════════════════════════════════════════════════════════════════════════════


# ── data loading ──────────────────────────────────────────────────────────────
def load_and_prepare(csv_path):
    # auto-detect encoding and separator
    try:
        import chardet
        with open(csv_path, "rb") as _f:
            enc = chardet.detect(_f.read(20000))["encoding"] or "latin-1"
    except ImportError:
        enc = "latin-1"
    with open(csv_path, "r", encoding=enc, errors="replace") as _f:
        first = _f.readline()
    sep = ";" if first.count(";") > first.count(",") else ","
    print(f"  Detected: encoding={enc}, separator='{sep}'")
    df_orig = pd.read_csv(csv_path, sep=sep, encoding=enc, encoding_errors="replace")
    df_orig.columns = df_orig.columns.str.strip()

    name_col = NAME_COL or df_orig.columns[0]
    sample_ids = df_orig[name_col].astype(str).tolist()

    if NUMERIC_COLS:
        feat_cols = [c for c in NUMERIC_COLS if c in df_orig.columns]
    else:
        feat_cols = [c for c in df_orig.columns
                     if c != name_col and c.lower() != "cluster"]

    X_raw = df_orig[feat_cols].apply(pd.to_numeric, errors="coerce")
    X_raw = X_raw.dropna(axis=1, how="all")
    X_raw = X_raw.fillna(X_raw.median())
    features = X_raw.columns.tolist()

    X_work = X_raw.copy()
    if LOG_TRANSFORM:
        for col in features:
            if col not in LOG_EXCLUDE:
                X_work[col] = np.log10(X_work[col].clip(lower=1e-10))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_work)

    return df_orig, name_col, sample_ids, features, X_raw, X_scaled, sep, enc


# ── helpers ───────────────────────────────────────────────────────────────────
def confidence_ellipse(x, y, ax, n_std=1.5, **kwargs):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1] + 1e-12)
    rx, ry = np.sqrt(1 + pearson), np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=rx * 2, height=ry * 2, **kwargs)
    sx = np.sqrt(cov[0, 0]) * n_std
    sy = np.sqrt(cov[1, 1]) * n_std
    t = (transforms.Affine2D()
         .rotate_deg(45).scale(sx, sy)
         .translate(np.mean(x), np.mean(y)))
    ellipse.set_transform(t + ax.transData)
    ax.add_patch(ellipse)


def cstyle(k):
    return PALETTE[(k - 1) % len(PALETTE)], MARKERS[(k - 1) % len(MARKERS)]


# ── plot: dendrogram ──────────────────────────────────────────────────────────
def plot_dendrogram(X_scaled, sample_ids, n_clusters, method, ax):
    Z = linkage(X_scaled, method=method)
    cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    leaf_col = {i: PALETTE[(cluster_labels[i] - 1) % len(PALETTE)]
                for i in range(len(sample_ids))}

    cut_h = Z[-(n_clusters - 1), 2]
    dendrogram(Z, labels=sample_ids, ax=ax,
               leaf_rotation=90, leaf_font_size=8,
               color_threshold=cut_h,
               link_color_func=lambda k: "#cccccc")

    for lbl in ax.get_xmajorticklabels():
        t = lbl.get_text()
        if t in sample_ids:
            lbl.set_color(leaf_col[sample_ids.index(t)])

    ax.axhline(cut_h, color="#D85A30", lw=1.2, ls="--",
               label=f"cut → {n_clusters} clusters")
    ax.set_ylabel("Distance", fontsize=11)
    ax.set_title(f"Hierarchical clustering  ({method} linkage)", fontsize=12, pad=8)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    return fcluster(Z, t=n_clusters, criterion="maxclust"), Z


# ── plot: PCA scores ──────────────────────────────────────────────────────────
def plot_scores(scores, var_exp, sample_ids, cluster_labels, n_clusters, ax):
    for k in range(1, n_clusters + 1):
        mask = cluster_labels == k
        c, m = cstyle(k)
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   color=c, marker=m, s=60, zorder=3,
                   label=f"Cluster {k}", edgecolors="white", linewidths=0.5)
        if mask.sum() >= 3:
            confidence_ellipse(scores[mask, 0], scores[mask, 1], ax,
                               alpha=0.12, facecolor=c, edgecolor=c, linewidth=1)

    for i, sid in enumerate(sample_ids):
        ax.annotate(sid, (scores[i, 0], scores[i, 1]),
                    fontsize=7.5, color="#555",
                    xytext=(4, 4), textcoords="offset points")

    ax.axhline(0, color="#ddd", lw=0.8, zorder=1)
    ax.axvline(0, color="#ddd", lw=0.8, zorder=1)
    ax.set_xlabel(f"PC1  ({var_exp[0]:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2  ({var_exp[1]:.1f}% variance)", fontsize=11)
    ax.set_title("PCA — scores", fontsize=12, pad=8)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


# ── plot: PCA loadings ────────────────────────────────────────────────────────
def plot_loadings(loadings, features, var_exp, ax):
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#e0e0e0", lw=0.8, zorder=1)

    mags = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    norm = (mags - mags.min()) / (mags.max() - mags.min() + 1e-9)
    cmap = plt.cm.Blues

    for j, feat in enumerate(features):
        lx, ly = loadings[j, 0], loadings[j, 1]
        col = cmap(0.4 + 0.55 * norm[j])
        ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=1.4, mutation_scale=11), zorder=3)
        nx = lx + 0.11 * np.sign(lx) if abs(lx) > 0.01 else lx
        ny = ly + 0.11 * np.sign(ly) if abs(ly) > 0.01 else ly
        ax.text(nx, ny, feat, fontsize=8.5, color="#185FA5",
                ha="center", va="center", fontweight="500",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.75))

    ax.axhline(0, color="#ddd", lw=0.7, zorder=1)
    ax.axvline(0, color="#ddd", lw=0.7, zorder=1)
    pad = 0.28
    ax.set_xlim(-1 - pad, 1 + pad)
    ax.set_ylim(-1 - pad, 1 + pad)
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC1  ({var_exp[0]:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2  ({var_exp[1]:.1f}%)", fontsize=11)
    ax.set_title("PCA — loadings", fontsize=12, pad=8)
    ax.spines[["top", "right"]].set_visible(False)


# ── plot: cluster profiles ────────────────────────────────────────────────────
def plot_profiles(X_raw, features, cluster_labels, n_clusters, ax):
    X_std = StandardScaler().fit_transform(X_raw)
    x = np.arange(len(features))
    for k in range(1, n_clusters + 1):
        mask = cluster_labels == k
        means = X_std[mask].mean(axis=0)
        stds  = X_std[mask].std(axis=0)
        c, _ = cstyle(k)
        ax.plot(x, means, color=c, lw=2, marker="o", ms=4,
                label=f"Cluster {k}  (n={mask.sum()})")
        ax.fill_between(x, means - stds, means + stds, color=c, alpha=0.09)

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="#ddd", lw=0.8)
    ax.set_ylabel("Standardised mean ± 1 SD", fontsize=11)
    ax.set_title("Cluster chemical profiles", fontsize=12, pad=8)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


# ── statistical summary ───────────────────────────────────────────────────────
def save_summary(csv_path, sample_ids, features, X_raw, X_scaled,
                 cluster_labels, Z, pca, scores_full, loadings_full, var_exp_full):
    # work with full-component arrays; slice to 2 only where needed for display
    scores   = scores_full[:, :2]
    loadings = loadings_full[:, :2]
    var_exp  = var_exp_full[:2]
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist

    out_txt = os.path.splitext(csv_path)[0] + "_summary.txt"
    base    = os.path.basename(csv_path)
    W       = 72   # line width for separators

    def sep(char="─"):
        return char * W

    lines = []
    a = lines.append

    # ── header ────────────────────────────────────────────────────────────────
    a(sep("═"))
    a("  GROUNDWATER MULTIVARIATE ANALYSIS  —  STATISTICAL SUMMARY")
    a(f"  Input file : {base}")
    a(f"  Samples    : {len(sample_ids)}")
    a(f"  Variables  : {len(features)}  →  {', '.join(features)}")
    a(f"  Log-transform : {'ON  (excluding: ' + ', '.join(LOG_EXCLUDE) + ')' if LOG_TRANSFORM else 'OFF'}")
    a(sep("═"))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — HCA
    # ══════════════════════════════════════════════════════════════════════════
    a("")
    a("  1.  HIERARCHICAL CLUSTER ANALYSIS (HCA)")
    a(sep())

    # settings
    cut_h = Z[-(N_CLUSTERS - 1), 2]
    a(f"  Linkage method  : {LINKAGE_METHOD}")
    a(f"  Number of clusters requested : {N_CLUSTERS}")
    a(f"  Dendrogram cut height        : {cut_h:.4f}")

    # cophenetic correlation
    c_coef, _ = cophenet(Z, pdist(X_scaled))
    a(f"  Cophenetic correlation coeff : {c_coef:.4f}"
      f"  {'(good fit ≥ 0.75)' if c_coef >= 0.75 else '(moderate fit — consider other linkage)'}")
    a("")

    # cluster membership & sizes
    a("  1a. Cluster membership")
    a(sep("·"))
    col_w = max(len(s) for s in sample_ids) + 2
    a(f"  {'Sample':<{col_w}}  Cluster")
    a(f"  {'------':<{col_w}}  -------")
    for sid, cl in zip(sample_ids, cluster_labels):
        a(f"  {sid:<{col_w}}  {cl}")

    a("")
    a("  1b. Cluster sizes")
    a(sep("·"))
    for k in range(1, N_CLUSTERS + 1):
        n = int((cluster_labels == k).sum())
        members = [s for s, c in zip(sample_ids, cluster_labels) if c == k]
        a(f"  Cluster {k} : n={n}  →  {', '.join(members)}")

    # inter-cluster distances (mean linkage distance from Z)
    a("")
    a("  1c. Inter-cluster distances  (from linkage matrix, last merges)")
    a(sep("·"))
    a("  The final merge distances from the linkage matrix Z (last N_CLUSTERS−1 rows):")
    a(f"  {'Step':<6}  {'Distance':>12}  {'Note'}")
    for i, row in enumerate(Z[-(N_CLUSTERS - 1):]):
        step = len(sample_ids) + i + 1
        a(f"  {step:<6}  {row[2]:>12.4f}  merges nodes {int(row[0])} and {int(row[1])}")

    # per-cluster mean ± SD in original units
    a("")
    a("  1d. Per-cluster chemical profiles  (original unscaled units, mean ± SD)")
    a(sep("·"))
    header = f"  {'Variable':<20}" + "".join(f"  {'Cluster '+str(k):>18}" for k in range(1, N_CLUSTERS + 1))
    a(header)
    a("  " + "─" * (len(header) - 2))
    X_arr = X_raw.values if hasattr(X_raw, "values") else X_raw
    for j, feat in enumerate(features):
        row_str = f"  {feat:<20}"
        for k in range(1, N_CLUSTERS + 1):
            mask = cluster_labels == k
            vals = X_arr[mask, j]
            row_str += f"  {np.mean(vals):>8.3f} ± {np.std(vals):>6.3f}"
        a(row_str)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — PCA
    # ══════════════════════════════════════════════════════════════════════════
    a("")
    a("")
    a("  2.  PRINCIPAL COMPONENT ANALYSIS (PCA)")
    a(sep())

    # variance explained
    a("  2a. Variance explained")
    a(sep("·"))
    cumvar = 0.0
    for i, ve in enumerate(var_exp):
        cumvar += ve
        a(f"  PC{i+1:>2} : {ve:>6.2f}%   (cumulative: {cumvar:>6.2f}%)")
    a("")
    a(f"  PC1 + PC2 combined : {var_exp[0]+var_exp[1]:.2f}%")

    # full PCA explained variance for ALL components
    if len(var_exp_full) > 2:
        a("")
        a("  All components:")
        cumvar = 0.0
        for i, ve in enumerate(var_exp_full):
            cumvar += ve
            a(f"    PC{i+1} : {ve:>6.2f}%   cumulative {cumvar:>6.2f}%")

    # loadings table
    a("")
    a("  2b. Loadings  (sorted by magnitude on PC1+PC2 plane)")
    a(sep("·"))
    ldf = pd.DataFrame(loadings, index=features, columns=["PC1", "PC2"])
    ldf["Magnitude"] = np.sqrt(ldf.PC1**2 + ldf.PC2**2)
    ldf_sorted = ldf.sort_values("Magnitude", ascending=False)
    a(f"  {'Variable':<20}  {'PC1':>8}  {'PC2':>8}  {'Magnitude':>10}")
    a(f"  {'--------':<20}  {'---':>8}  {'---':>8}  {'---------':>10}")
    for feat, row in ldf_sorted.iterrows():
        a(f"  {feat:<20}  {row.PC1:>8.4f}  {row.PC2:>8.4f}  {row.Magnitude:>10.4f}")

    # PC scores per sample
    a("")
    a("  2c. PC scores per sample")
    a(sep("·"))
    a(f"  {'Sample':<{col_w}}  {'Cluster':>7}  {'PC1':>10}  {'PC2':>10}")
    a(f"  {'------':<{col_w}}  {'-------':>7}  {'---':>10}  {'---':>10}")
    for i, sid in enumerate(sample_ids):
        a(f"  {sid:<{col_w}}  {cluster_labels[i]:>7}  {scores[i,0]:>10.4f}  {scores[i,1]:>10.4f}")

    # ── footer ────────────────────────────────────────────────────────────────
    a("")
    a(sep("═"))
    a("  END OF SUMMARY")
    a(sep("═"))

    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Summary saved        → {out_txt}")


# ── main ──────────────────────────────────────────────────────────────────────
def main(csv_path):
    print(f"\nLoading: {csv_path}")
    df_orig, name_col, sample_ids, features, X_raw, X_scaled, sep, enc = \
        load_and_prepare(csv_path)
    print(f"  {len(sample_ids)} samples  |  {len(features)} variables")
    print(f"  Variables: {features}")
    if LOG_TRANSFORM:
        print(f"  Log-transform ON  (excluding: {LOG_EXCLUDE})")

    # ── HCA ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.48, wspace=0.36,
                           left=0.07, right=0.97, top=0.93, bottom=0.10)

    ax_dend  = fig.add_subplot(gs[0, :])
    ax_score = fig.add_subplot(gs[1, 0])
    ax_load  = fig.add_subplot(gs[1, 1])
    ax_prof  = fig.add_subplot(gs[1, 2])

    cluster_labels, Z = plot_dendrogram(
        X_scaled, sample_ids, N_CLUSTERS, LINKAGE_METHOD, ax_dend)

    # ── PCA ──────────────────────────────────────────────────────────────────
    # Fit on ALL components so eigenvectors and explained variance are exact;
    # slice to 2 components only for plotting.
    pca           = PCA(n_components=X_scaled.shape[1])
    scores_full   = pca.fit_transform(X_scaled)
    loadings_full = pca.components_.T          # shape: (n_features, n_components)
    var_exp_full  = pca.explained_variance_ratio_ * 100

    scores   = scores_full[:, :2]
    loadings = loadings_full[:, :2]
    var_exp  = var_exp_full[:2]

    plot_scores(scores, var_exp, sample_ids, cluster_labels, N_CLUSTERS, ax_score)
    plot_loadings(loadings, features, var_exp, ax_load)
    plot_profiles(X_raw.values, features, cluster_labels, N_CLUSTERS, ax_prof)

    base = os.path.basename(csv_path)
    fig.suptitle(f"Groundwater multivariate analysis  —  {base}",
                 fontsize=14, fontweight="500", y=0.97)

    out_fig = os.path.splitext(csv_path)[0] + "_multivariate.png"
    fig.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_fig}")

    # ── append Cluster column back into the original CSV (in-place) ─────────
    df_out = df_orig.copy()
    if "Cluster" in df_out.columns:
        df_out = df_out.drop(columns=["Cluster"])
    name_pos = df_out.columns.get_loc(name_col)
    df_out.insert(name_pos + 1, "Cluster", cluster_labels)
    df_out.to_csv(csv_path, index=False, sep=sep, encoding=enc)
    print(f"Cluster column written back → {csv_path}  (sep='{sep}', enc={enc})")

    # ── print summary table ───────────────────────────────────────────────────
    print(f"\n{'Sample':<22} {'Cluster':>7}")
    print("─" * 30)
    for sid, cl in zip(sample_ids, cluster_labels):
        print(f"  {sid:<20} {cl:>7}")

    print(f"\nVariance explained:  PC1={var_exp[0]:.1f}%  PC2={var_exp[1]:.1f}%  "
          f"Total={var_exp[0]+var_exp[1]:.1f}%")

    ldf = pd.DataFrame(loadings, index=features, columns=["PC1", "PC2"])
    ldf["magnitude"] = np.sqrt(ldf.PC1**2 + ldf.PC2**2)
    print("\nLoadings (sorted by magnitude):")
    print(ldf.sort_values("magnitude", ascending=False).round(3).to_string())

    # ── save statistical summary ──────────────────────────────────────────────
    save_summary(csv_path, sample_ids, features, X_raw, X_scaled,
                 cluster_labels, Z, pca, scores_full, loadings_full, var_exp_full)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print(f"\nCurrent settings:")
        print(f"  N_CLUSTERS     = {N_CLUSTERS}")
        print(f"  LINKAGE_METHOD = {LINKAGE_METHOD}")
        print(f"  LOG_TRANSFORM  = {LOG_TRANSFORM}")
        sys.exit(0)
    main(sys.argv[1])
