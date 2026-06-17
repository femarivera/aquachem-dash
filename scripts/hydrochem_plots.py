"""
hydrochem_plots.py
──────────────────
Single-CSV input

Usage:
    python hydrochem_plots.py  your_data_with_clusters.csv

Options:
    --symbology  CLUSTER|AQUIFER   Colour/marker scheme (default: AQUIFER)
    --label-column  COLUMN_NAME    Column to use as point labels when
                                   SHOW_LABELS=True (default: auto-detect
                                   from NAME_CANDIDATES)
    --filter-column  COLUMN_NAME   Column to filter rows by (requires
                                   --filter-value)
    --filter-value  VALUE [VALUE …] One or more values to keep; rows whose
                                   FILTER_COLUMN does not match are excluded
                                   from every plot.
                                   Example: --filter-column Aquifer
                                            --filter-value Miocene Eocene
"""

import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines
from matplotlib.patches import Polygon as MPoly
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE
# ══════════════════════════════════════════════════════════════════════════════
PALETTE      = ["#185FA5","#D85A30","#1D9E75","#BA7517","#993556","#534AB7","#3B6D11"]
MARKERS      = ["o","^","s","D","v","P","X"]
MS           = 62     # marker size
DPI          = 300
SHOW_LABELS  = True   # set True to annotate each point with a label
LABEL_COLUMN = "Flowpath order"  # column to use as labels when SHOW_LABELS=True;
                                  # set to None to auto-detect from NAME_CANDIDATES
SYMBOLOGY    = "AQUIFER"   # "CLUSTER" or "AQUIFER"

# ── Row filtering ──────────────────────────────────────────────────────────────
# When FILTER=True, only rows whose FILTER_COLUMN value matches one of the
# entries in FILTER_VALUE will be included in every plot.
# FILTER_VALUE accepts a single value  : FILTER_VALUE = "Miocene"
#               or a list of values    : FILTER_VALUE = ["Miocene", "Eocene"]
FILTER        = False         # set True to activate row filtering
FILTER_COLUMN = "Aquifer"          # column name string, e.g. "Aquifer"
FILTER_VALUE  = "Oligocene"          # single value or list, e.g. ["Miocene", "Eocene"]
H        = np.sqrt(3)/2   # equilateral triangle height (side=1)
GAP      = 0.20           # gap between triangles in Piper

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.labelsize": 9,     "ytick.labelsize": 9,
})

def cstyle(k):
    i = int(k) - 1
    return PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]

def cl_legend(ax, clusters, loc="best"):
    h = [mlines.Line2D([],[],color=cstyle(k)[0],marker=cstyle(k)[1],
                       ls="None",ms=6,label=f"Cluster {int(k)}")
         for k in sorted(clusters.dropna().unique())]
    ax.legend(handles=h,fontsize=8,frameon=True,
              framealpha=0.92,edgecolor="#ddd",loc=loc)

# ── Aquifer symbology ─────────────────────────────────────────────────────────
# Canonical aquifer names → (colour, display label)
AQUIFER_PALETTE = {
    "PlioQ":     ("#555555", "PlioQ"),        # dark grey
    "Miocene":   ("#E0CA00", "Miocene"),      # dark yellow
    "Eocene":    ("#E86000", "Eocene"),       # orange
    "Oligocene": ("#6A0DAD", "Oligocene"),    # purple
}
# All recognised aliases for each canonical name (lower-case)
_AQ_ALIASES = {
    "PlioQ":     {"plioq", "plioquaternary", "plioquaternay", "plio-quaternary",
                  "plio quaternary", "plioquet", "plioqu"},
    "Miocene":   {"miocene", "mio"},
    "Eocene":    {"eocene", "eoc"},
    "Oligocene": {"oligocene", "oligocen", "olig", "oli"},
}
# Build a flat lookup: alias_lower → canonical
_AQ_LOOKUP = {alias: canon
              for canon, aliases in _AQ_ALIASES.items()
              for alias in aliases}

def normalize_aquifer(val):
    """Map a raw Aquifer cell value to a canonical name (or 'Unknown')."""
    if pd.isna(val):
        return "Unknown"
    return _AQ_LOOKUP.get(str(val).strip().lower(), str(val).strip())

def aquifer_style(aq):
    """Return (colour, marker) for a canonical aquifer name."""
    colour = AQUIFER_PALETTE.get(aq, ("#888888", "Unknown"))[0]
    # Assign a consistent marker by position in the palette list
    idx = list(AQUIFER_PALETTE.keys()).index(aq) if aq in AQUIFER_PALETTE else len(AQUIFER_PALETTE)
    return colour, MARKERS[idx % len(MARKERS)]

def aq_legend(ax, aquifers, loc="best"):
    """Legend helper for AQUIFER symbology mode."""
    seen = sorted(aquifers.dropna().unique())
    h = [mlines.Line2D([],[],
                       color=aquifer_style(a)[0],
                       marker=aquifer_style(a)[1],
                       ls="None", ms=6,
                       label=AQUIFER_PALETTE.get(a, (None, a))[1])
         for a in seen]
    ax.legend(handles=h, fontsize=8, frameon=True,
              framealpha=0.92, edgecolor="#ddd", loc=loc)

def scat(ax, x, y, cl, names=None, aquifer=None, **kw):
    if SYMBOLOGY == "AQUIFER" and aquifer is not None:
        for a in sorted(aquifer.dropna().unique()):
            c, m = aquifer_style(a)
            mask = aquifer == a
            ax.scatter(x[mask], y[mask], color=c, marker=m, s=MS,
                       zorder=3, edgecolors="white", linewidths=0.4, **kw)
    else:
        for k in sorted(cl.dropna().unique()):
            c,m = cstyle(k)
            mask = cl==k
            ax.scatter(x[mask],y[mask],color=c,marker=m,s=MS,
                       zorder=3,edgecolors="white",linewidths=0.4,**kw)
    if names is not None and SHOW_LABELS:
        for xi, yi, n in zip(x.values, y.values, names.values):
            if pd.isna(xi) or pd.isna(yi): continue
            ax.annotate(str(n), (xi, yi),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=6.5, color="#222", fontweight="500",
                        zorder=6, clip_on=True,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec="none", alpha=0.7))

def style(ax, xl="", yl="", title=""):
    ax.set_xlabel(xl,fontsize=10); ax.set_ylabel(yl,fontsize=10)
    ax.set_title(title,fontsize=11,fontweight="600",pad=8)
    ax.spines[["top","right"]].set_visible(False)

def save(fig, name, out_dir):
    p = os.path.join(out_dir, name+".png")
    fig.savefig(p,dpi=DPI,bbox_inches="tight"); plt.close(fig)
    print(f"  saved → {p}")

# ══════════════════════════════════════════════════════════════════════════════
#  MEQ / MMOL
# ══════════════════════════════════════════════════════════════════════════════
MW  = dict(Na=22.990,K=39.098,Mg=24.305,Ca=40.078,
           Cl=35.453,HCO3=61.016,SO4=96.06,NH4=18.038,F=18.998,NO3=62.004)
VAL = dict(Na=1,K=1,Mg=2,Ca=2,Cl=1,HCO3=1,SO4=2,NH4=1,F=1,NO3=1)
meq  = lambda s,i: pd.to_numeric(s,errors="coerce")*VAL[i]/MW[i]
mmol = lambda s,i: pd.to_numeric(s,errors="coerce")/MW[i]

LOOKUPS = {
    "Na":   ["na (ppm)","na"],
    "K":    ["k (ppm)","k"],
    "Mg":   ["mg (ppm)","mg"],
    "Ca":   ["ca (ppm)","ca"],
    "Cl":   ["cl (ppm)","cl"],
    "HCO3": ["hco3 (ppm)","hco3"],
    "SO4":  ["so4 (ppm)","so4"],
    "NH4":  ["nh4 (ppm)","nh4"],
    "F":    ["f (ppm)","f"],
    "NO3":  ["no3 (ppm)","no3"],
    "pH":   ["ph"],
    "T":    ["t","t (°c)","t (oc)","temp","temperature"],
    "CE":   ["ce","ce (µs/cm)","ce (us/cm)","ec","conductivity"],
    "DO":   ["do","do2","do2 (%)","do (%)","dissolved oxygen"],
    "C14":  ["14c (pmc)","14c","c14"],
    "C14s": ["14c stdev","14c std"],
    "C13":  ["d13c","δ13c"],
    "C13s": ["d13c stdev","d13c std"],
    "D2H":  ["d2h","δ2h","d2h (‰)"],
    "D2Hs": ["d2h stdev","d2h std"],
    "D18O": ["d18o","δ18o","d18o (‰)"],
    "D18Os":["d18o stdev","d18o std"],
    "PCO2": ["p_co2(g)","p_co2","pco2","log pco2"],
    "siCal":["si_calcite","si calcite"],
    "siDol":["si_dolomite","si dolomite"],
    "siGyp":["si_gypsum","si gypsum"],
    "siF":  ["si_fluorite","si fluorite"],
    "SAR":  ["sar"],
    "TDS":  ["tds","total dissolved solids","tds (mg/l)","tds (ppm)",
              "tds (mg/l)","mineralization","mineralisation"],
    "Li":   ["li (ppm)","li"],
    "Br":   ["br (ppm)","br"],
    "PO4":  ["po4 (ppm)","po4"],
    "And":  ["And_Netpath","And Netpath","and_netpath","and netpath"],
    "C13c": ["13c_computed","d13c_computed","13c computed","d13c computed"],
    "Acorr":["Acorr_Netpath","Acorr Netpath","acorr_netpath","acorr netpath"],
    "AndP": ["And_Pearson","And Pearson","and_pearson","and pearson"],
    "AcorrP":["Acorr_Pearson", "Acorr Pearson","acorr_pearson","acorr pearson"],
    "AgeNP":["Age_Netpath","Age Netpath","age_netpath","age netpath"],
    "AgePR":["Age_Pearson","Age Pearson","age_pearson","age pearson"],
    "RatioPNP":["ratio_pearson_netpath","ratio pearson netpath"],
    "Borehole":["borehole","well","well name","bore"],
}

def fcol(df, key):
    low = {c.lower().strip():c for c in df.columns}
    for cand in LOOKUPS.get(key,[key.lower()]):
        if cand in low: return low[cand]
    return None

# Name column candidates — first match wins (case-insensitive)
NAME_CANDIDATES = ["name", "sample", "sample name", "sample id",
                   "sampleid", "id", "code", "borehole", "well"]

def load(path):
    # auto-detect encoding (handles latin-1 / cp1252 from European Excel exports)
    try:
        import chardet
        with open(path, "rb") as _f:
            enc = chardet.detect(_f.read(20000))["encoding"] or "latin-1"
    except ImportError:
        enc = "latin-1"   # safe fallback — covers °, µ, accents
    # sniff separator: semicolon (European CSV) vs comma
    with open(path, "r", encoding=enc, errors="replace") as _f:
        first = _f.readline()
    sep = ";" if first.count(";") > first.count(",") else ","
    df = pd.read_csv(path, sep=sep, encoding=enc, encoding_errors="replace")
    df.columns = df.columns.str.strip()
    # Build the label column used by SHOW_LABELS annotations.
    # LABEL_COLUMN is the single source of truth — no auto-detect fallback.
    if LABEL_COLUMN is not None:
        lc = LABEL_COLUMN.strip()
        col_map = {c.lower().strip(): c for c in df.columns}
        if lc in df.columns:
            df["Name"] = df[lc].values
        elif lc.lower() in col_map:
            df["Name"] = df[col_map[lc.lower()]].values
        else:
            raise ValueError(
                f"LABEL_COLUMN='{LABEL_COLUMN}' not found in CSV.\n"
                f"  Available columns: {list(df.columns)}"
            )
        print(f"  Label column: '{lc}'")
    else:
        df["Name"] = df.iloc[:, 0].values
        print(f"  Label column: '{df.columns[0]}' (first column, auto)")
    cc = next((c for c in df.columns if c.lower()=="cluster"),None)
    if cc:
        df["Cluster"] = pd.to_numeric(df[cc],errors="coerce").fillna(1).astype(int)
    else:
        print("  WARNING: no Cluster column — assigning all to cluster 1")
        df["Cluster"] = 1
    COL = {k:fcol(df,k) for k in LOOKUPS}
    for ion in ["Na","K","Mg","Ca","Cl","HCO3","SO4","NH4","F","NO3"]:
        if COL[ion]:
            df[f"{ion}_meq"]  = meq(df[COL[ion]],ion)
            df[f"{ion}_mmol"] = mmol(df[COL[ion]],ion)
    # Parse Aquifer column for AQUIFER symbology
    aq_raw = next((c for c in df.columns if c.strip().lower() == "aquifer"), None)
    if aq_raw:
        df["Aquifer"] = df[aq_raw].apply(normalize_aquifer)
        print(f"  Aquifer column: '{aq_raw}'  →  {sorted(df['Aquifer'].unique())}")
    else:
        df["Aquifer"] = "Unknown"
        if SYMBOLOGY == "AQUIFER":
            print("  WARNING: no Aquifer column found — all points labelled 'Unknown'")
    return df, COL

def apply_filter(df):
    """Return a filtered copy of *df* when FILTER=True, otherwise return df unchanged."""
    if not FILTER:
        return df
    if not FILTER_COLUMN:
        print("  WARNING: FILTER=True but FILTER_COLUMN is not set — filter ignored")
        return df
    if FILTER_VALUE is None:
        print("  WARNING: FILTER=True but FILTER_VALUE is not set — filter ignored")
        return df

    # Resolve the column name (case-insensitive, strip whitespace)
    col_map = {c.strip().lower(): c for c in df.columns}
    resolved = None
    if FILTER_COLUMN in df.columns:
        resolved = FILTER_COLUMN
    elif FILTER_COLUMN.strip().lower() in col_map:
        resolved = col_map[FILTER_COLUMN.strip().lower()]
    else:
        raise ValueError(
            f"FILTER_COLUMN='{FILTER_COLUMN}' not found in CSV.\n"
            f"  Available columns: {list(df.columns)}"
        )

    # Normalise FILTER_VALUE to a list for uniform handling
    values = [FILTER_VALUE] if not isinstance(FILTER_VALUE, (list, tuple, set)) else list(FILTER_VALUE)

    mask = df[resolved].astype(str).str.strip().isin([str(v).strip() for v in values])
    filtered = df[mask].copy()
    n_before, n_after = len(df), len(filtered)
    print(f"  Filter: '{resolved}' ∈ {values}  →  {n_before} → {n_after} rows kept")
    if filtered.empty:
        print("  WARNING: filter matched 0 rows — all plots will be skipped")
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — CAI
# ══════════════════════════════════════════════════════════════════════════════
def plot_cai(df, out):
    req=["Na_meq","K_meq","Cl_meq","SO4_meq","HCO3_meq"]
    d=df.dropna(subset=req).copy()
    if d.empty: print("  CAI skipped"); return
    num  = d["Cl_meq"]-(d["Na_meq"]+d["K_meq"])
    cai1 = num/d["Cl_meq"].replace(0,np.nan)
    cai2 = num/(d["SO4_meq"]+d["HCO3_meq"]).replace(0,np.nan)
    fig,ax=plt.subplots(figsize=(6,5))
    scat(ax,cai1,cai2,d["Cluster"],names=d["Name"],aquifer=d["Aquifer"])
    ax.axhline(0,color="#999",lw=1.0,ls="--"); ax.axvline(0,color="#999",lw=1.0,ls="--")
    if SYMBOLOGY == "AQUIFER":
        aq_legend(ax, d["Aquifer"])
    else:
        cl_legend(ax,d["Cluster"])
    style(ax,"CAI-1","CAI-2","Chloro-Alkaline Indices")
    fig.tight_layout(); save(fig,"01_CAI",out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — (Ca²⁺ + Mg²⁺) vs HCO₃⁻  (Na-normalised, meq/L)
# ══════════════════════════════════════════════════════════════════════════════
def plot_na_norm(df, out):
    req=["Na_meq","Ca_meq","Mg_meq","HCO3_meq"]
    d=df.dropna(subset=req).copy()
    if d.empty: print("  Na-norm skipped"); return
    x=(d["Ca_meq"]+d["Mg_meq"])/d["Na_meq"].replace(0,np.nan)
    y=d["HCO3_meq"]/d["Na_meq"].replace(0,np.nan)
    fig,ax=plt.subplots(figsize=(6,5))
    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    lims = [0, max(x.max(), y.max()) * 1.05]
    ax.plot(lims, lims, color="#aaa", lw=1.0, ls="--", zorder=1, label="1:1")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    handles_extra = [mlines.Line2D([],[],color="#aaa",ls="--",lw=1,label="1:1")]
    if SYMBOLOGY == "AQUIFER":
        for a in sorted(d["Aquifer"].unique()):
            c,m=aquifer_style(a); handles_extra.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=AQUIFER_PALETTE.get(a,(None,a))[1]))
    else:
        for k in sorted(d["Cluster"].unique()):
            c,m=cstyle(k); handles_extra.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=f"Cluster {int(k)}"))
    ax.legend(handles=handles_extra,fontsize=8,frameon=True,framealpha=0.92,edgecolor="#ddd")
    style(ax,"(Ca²⁺+Mg²⁺)/Na⁺  (meq/L)","HCO₃⁻/Na⁺  (meq/L)","(Ca²⁺ + Mg²⁺) vs HCO₃⁻  (Na-normalised, meq/L)")
    fig.tight_layout(); save(fig,"02_HCO3_vs_CaMg_norm",out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — (Ca²⁺ + Mg²⁺) vs HCO₃⁻  (unnormalised, meq/L)
# ══════════════════════════════════════════════════════════════════════════════
def plot_hco3_camg(df, out):
    req=["Ca_meq","Mg_meq","HCO3_meq"]
    d=df.dropna(subset=req).copy()
    if d.empty: print("  HCO3 vs Ca+Mg skipped"); return
    x = d["Ca_meq"] + d["Mg_meq"]
    y = d["HCO3_meq"]
    fig,ax=plt.subplots(figsize=(6,5))
    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    lims = [0, max(x.max(), y.max()) * 1.08]
    ax.plot(lims, lims, color="#aaa", lw=1.0, ls="--", zorder=1)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    handles_extra = [mlines.Line2D([],[],color="#aaa",ls="--",lw=1,label="1:1")]
    if SYMBOLOGY == "AQUIFER":
        for a in sorted(d["Aquifer"].unique()):
            c,m=aquifer_style(a); handles_extra.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=AQUIFER_PALETTE.get(a,(None,a))[1]))
    else:
        for k in sorted(d["Cluster"].unique()):
            c,m=cstyle(k); handles_extra.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=f"Cluster {int(k)}"))
    ax.legend(handles=handles_extra,fontsize=8,frameon=True,framealpha=0.92,edgecolor="#ddd")
    style(ax,"Ca²⁺ + Mg²⁺  (meq/L)","HCO₃⁻  (meq/L)","(Ca²⁺ + Mg²⁺) vs HCO₃⁻  (unnormalised, meq/L)")
    fig.tight_layout(); save(fig,"03_HCO3_vs_CaMg",out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 4 — Cl vs Na/Cl
# ══════════════════════════════════════════════════════════════════════════════
def plot_nacl(df, out):
    req=["Na_meq","Cl_meq"]
    d=df.dropna(subset=req).copy()
    if d.empty: print("  Na/Cl skipped"); return
    x=d["Cl_meq"]; y=d["Na_meq"]/d["Cl_meq"].replace(0,np.nan)
    fig,ax=plt.subplots(figsize=(6,5))
    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    ax.axhline(1,color="#555",lw=1.0,ls="--",zorder=1)
    h=[mlines.Line2D([],[],color="#555",ls="--",lw=1,label="Na/Cl=1")]
    if SYMBOLOGY == "AQUIFER":
        for a in sorted(d["Aquifer"].unique()):
            c,m=aquifer_style(a); h.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=AQUIFER_PALETTE.get(a,(None,a))[1]))
    else:
        for k in sorted(d["Cluster"].unique()):
            c,m=cstyle(k); h.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=f"Cluster {int(k)}"))
    ax.legend(handles=h,fontsize=8,frameon=True,framealpha=0.92,edgecolor="#ddd")
    style(ax,"Cl⁻  (meq/L)","Na⁺/Cl⁻  (meq/L)","Cl vs Na/Cl")
    fig.tight_layout(); save(fig,"04_Cl_vs_NaCl",out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 5 — Cl vs Ca/Cl
# ══════════════════════════════════════════════════════════════════════════════
def plot_cacl(df, out):
    req=["Ca_meq","Cl_meq"]
    d=df.dropna(subset=req).copy()
    if d.empty: print("  Ca/Cl skipped"); return
    x=d["Cl_meq"]; y=d["Ca_meq"]/d["Cl_meq"].replace(0,np.nan)
    fig,ax=plt.subplots(figsize=(6,5))
    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    if SYMBOLOGY == "AQUIFER":
        aq_legend(ax, d["Aquifer"])
    else:
        cl_legend(ax,d["Cluster"])
    style(ax,"Cl⁻  (meq/L)","Ca²⁺/Cl⁻  (meq/L)","Cl vs Ca/Cl")
    fig.tight_layout(); save(fig,"05_Cl_vs_CaCl",out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 6 — TDS vs EC
# ══════════════════════════════════════════════════════════════════════════════
def plot_ec_tds(df, COL, out):
    ce_col  = COL.get("CE")
    tds_col = COL.get("TDS")
    if not ce_col or not tds_col or tds_col not in df.columns:
        print("  EC vs TDS skipped (columns missing)"); return
    d = df.dropna(subset=[ce_col, tds_col]).copy()
    d["CE_num"]  = pd.to_numeric(d[ce_col],  errors="coerce")
    d["TDS_num"] = pd.to_numeric(d[tds_col], errors="coerce")
    d = d.dropna(subset=["CE_num","TDS_num"])
    if d.empty: print("  EC vs TDS skipped (no data)"); return
    fig,ax=plt.subplots(figsize=(6,5))
    scat(ax, d["TDS_num"], d["CE_num"], d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    if SYMBOLOGY == "AQUIFER":
        aq_legend(ax, d["Aquifer"])
    else:
        cl_legend(ax, d["Cluster"])
    style(ax, "TDS  (mg/L)", "EC  (µS/cm)", "TDS vs EC")
    fig.tight_layout(); save(fig,"06_TDS_vs_EC",out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 7 — (Na+K)−Cl  vs  (Ca+Mg)−(HCO3+SO4)  — cation exchange diagnostic
# ══════════════════════════════════════════════════════════════════════════════
def plot_exchange_balance(df, out):
    req = ["Na_meq","K_meq","Cl_meq","Ca_meq","Mg_meq","HCO3_meq","SO4_meq"]
    d = df.dropna(subset=req).copy()
    if d.empty: print("  Exchange balance skipped"); return
    x = (d["Na_meq"] + d["K_meq"]) - d["Cl_meq"]
    y = (d["Ca_meq"] + d["Mg_meq"]) - (d["HCO3_meq"] + d["SO4_meq"])
    fig, ax = plt.subplots(figsize=(6, 5))
    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    # slope -1 reference line through origin
    lim = max(abs(x).max(), abs(y).max()) * 1.15
    ref = np.array([-lim, lim])
    ax.plot(ref, -ref, color="black", lw=1.0, ls="--", zorder=1)
    ax.axhline(0, color="#ddd", lw=0.7, zorder=0)
    ax.axvline(0, color="#ddd", lw=0.7, zorder=0)
    handles_extra = [mlines.Line2D([],[],color="black",ls="--",lw=1,label="slope −1")]
    if SYMBOLOGY == "AQUIFER":
        for a in sorted(d["Aquifer"].unique()):
            c, m = aquifer_style(a)
            handles_extra.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=AQUIFER_PALETTE.get(a,(None,a))[1]))
    else:
        for k in sorted(d["Cluster"].unique()):
            c, m = cstyle(k)
            handles_extra.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=f"Cluster {int(k)}"))
    ax.legend(handles=handles_extra, fontsize=8, frameon=True, framealpha=0.92, edgecolor="#ddd")
    style(ax, "(Na⁺+K⁺) − Cl⁻  (meq/L)",
              "(Ca²⁺+Mg²⁺) − (HCO₃⁻+SO₄²⁻)  (meq/L)",
              "Cation exchange balance")
    fig.tight_layout(); save(fig, "07_Exchange_balance", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 8 — pH vs HCO₃⁻  (mmol/L)
# ══════════════════════════════════════════════════════════════════════════════
def plot_hco3_vs_ph(df, COL, out):
    ph_col  = COL.get("pH")
    hco_col = COL.get("HCO3")
    if not ph_col or not hco_col or ph_col not in df.columns or hco_col not in df.columns:
        print("  pH vs HCO₃⁻ skipped (columns missing)"); return
    d = df.dropna(subset=[ph_col, hco_col]).copy()
    if d.empty: print("  pH vs HCO₃⁻ skipped (no data)"); return

    x = pd.to_numeric(d[ph_col],  errors="coerce")
    y = pd.to_numeric(d[hco_col], errors="coerce")/MW["HCO3"]

    fig, ax = plt.subplots(figsize=(6, 5))
    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    if SYMBOLOGY == "AQUIFER":
        aq_legend(ax, d["Aquifer"])
    else:
        cl_legend(ax, d["Cluster"])
    style(ax, "pH", "HCO₃  (mmol/L)", "pH vs HCO₃⁻")
    fig.tight_layout()
    save(fig, "08_pH_vs_HCO3", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 9 — Cl⁻ vs Na⁺ (meq/L)
# ══════════════════════════════════════════════════════════════════════════════
def plot_na_vs_cl(df, out):
    req = ["Na_meq", "Cl_meq"]
    d = df.dropna(subset=req).copy()
    if d.empty: print("  Na vs Cl skipped"); return
    x = d["Cl_meq"]; y = d["Na_meq"]
    fig, ax = plt.subplots(figsize=(6, 5))
    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    lims = [0, max(x.max(), y.max()) * 1.08]
    ax.plot(lims, lims, color="#aaa", lw=1.0, ls="--", zorder=1)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    handles_extra = [mlines.Line2D([],[],color="#aaa",ls="--",lw=1,label="1:1")]
    if SYMBOLOGY == "AQUIFER":
        for a in sorted(d["Aquifer"].unique()):
            c, m = aquifer_style(a)
            handles_extra.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=AQUIFER_PALETTE.get(a,(None,a))[1]))
    else:
        for k in sorted(d["Cluster"].unique()):
            c, m = cstyle(k)
            handles_extra.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=f"Cluster {int(k)}"))
    ax.legend(handles=handles_extra, fontsize=8, frameon=True, framealpha=0.92, edgecolor="#ddd")
    style(ax, "Cl⁻  (meq/L)", "Na⁺  (meq/L)", "Cl⁻ vs Na⁺")
    fig.tight_layout(); save(fig, "09_Cl_vs_Na", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 10 — Cl⁻ vs Ca²⁺ (meq/L)
# ══════════════════════════════════════════════════════════════════════════════
def plot_ca_vs_cl(df, out):
    req = ["Ca_meq", "Cl_meq"]
    d = df.dropna(subset=req).copy()
    if d.empty: print("  Ca vs Cl skipped"); return
    x = d["Cl_meq"]; y = d["Ca_meq"]
    fig, ax = plt.subplots(figsize=(6, 5))
    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    if SYMBOLOGY == "AQUIFER":
        aq_legend(ax, d["Aquifer"])
    else:
        cl_legend(ax, d["Cluster"])
    style(ax, "Cl⁻  (meq/L)", "Ca²⁺  (meq/L)", "Cl⁻ vs Ca²⁺")
    fig.tight_layout(); save(fig, "10_Cl_vs_Ca", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 11 — Cl⁻ vs SO₄²⁻ (meq/L)
# ══════════════════════════════════════════════════════════════════════════════
def plot_so4_vs_cl(df, out):
    req = ["SO4_meq", "Cl_meq"]
    d = df.dropna(subset=req).copy()
    if d.empty: print("  SO4 vs Cl skipped"); return
    x = d["Cl_meq"]; y = d["SO4_meq"]
    fig, ax = plt.subplots(figsize=(6, 5))
    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    if SYMBOLOGY == "AQUIFER":
        aq_legend(ax, d["Aquifer"])
    else:
        cl_legend(ax, d["Cluster"])
    style(ax, "Cl⁻ (meq/L)", "SO₄²⁻  (meq/L)", "Cl⁻ vs SO₄²⁻")
    fig.tight_layout(); save(fig, "11_Cl_vs_SO4", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 12 — SO₄²⁻ vs Ca²⁺ (meq/L)
# ══════════════════════════════════════════════════════════════════════════════
def plot_ca_vs_so4(df, out):
    req = ["Ca_meq", "SO4_meq"]
    d = df.dropna(subset=req).copy()
    if d.empty: print("  Ca vs SO4 skipped"); return
    x = d["SO4_meq"]; y = d["Ca_meq"]
    fig, ax = plt.subplots(figsize=(6, 5))
    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    lims = [0, max(x.max(), y.max()) * 1.08]
    ax.plot(lims, lims, color="#aaa", lw=1.0, ls="--", zorder=1)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    handles_extra = [mlines.Line2D([],[],color="#aaa",ls="--",lw=1,label="1:1")]
    if SYMBOLOGY == "AQUIFER":
        for a in sorted(d["Aquifer"].unique()):
            c, m = aquifer_style(a)
            handles_extra.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=AQUIFER_PALETTE.get(a,(None,a))[1]))
    else:
        for k in sorted(d["Cluster"].unique()):
            c, m = cstyle(k)
            handles_extra.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=f"Cluster {int(k)}"))
    ax.legend(handles=handles_extra, fontsize=8, frameon=True, framealpha=0.92, edgecolor="#ddd")
    style(ax, "SO₄²⁻  (meq/L)", "Ca²⁺  (meq/L)", "SO₄²⁻ vs Ca²⁺")
    fig.tight_layout(); save(fig, "12_SO4_vs_Ca", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 13 — log pCO2 vs HCO₃⁻ and Ca²⁺
# ══════════════════════════════════════════════════════════════════════════════
def plot_pco2(df, COL, out):
    pc=COL.get("PCO2")
    if not pc or pc not in df.columns: print("  pCO2 skipped"); return
    req=["Ca_mmol","HCO3_mmol"]
    d=df.dropna(subset=req+[pc]).copy()
    if d.empty: print("  pCO2 skipped (no data)"); return
    logp=pd.to_numeric(d[pc],errors="coerce")
    # PHREEQC open-system calcite dissolution at 15°C (Ca and Alk in mmol/L)
    pco2_ref = [-1.5, -2.0, -2.5, -3.0, -3.5]
    ca_ref   = [2.96, 1.95, 1.29, 0.86, 0.58]
    alk_ref  = [5.79, 3.82, 2.53, 1.68, 1.10]

    fig,ax=plt.subplots(figsize=(7,5.5))
    ax.plot(pco2_ref, ca_ref,  color="black", lw=1.6, ls="-",
            marker="o", ms=4, label="Ca²⁺ open system 15°C (PHREEQC)")
    ax.plot(pco2_ref, alk_ref, color="black", lw=1.6, ls="--",
            marker="s", ms=4, label="HCO₃⁻ open system 15°C (PHREEQC)")
    _sym_groups = d["Aquifer"].unique() if SYMBOLOGY == "AQUIFER" else sorted(d["Cluster"].unique())
    for grp in _sym_groups:
        c,m = (aquifer_style(grp) if SYMBOLOGY == "AQUIFER" else cstyle(grp))
        mask = (d["Aquifer"]==grp) if SYMBOLOGY == "AQUIFER" else (d["Cluster"]==grp)
        ax.scatter(logp[mask],d["Ca_mmol"][mask],
                   color=c,marker=m,s=MS,zorder=4,edgecolors="white",linewidths=0.4)
        ax.scatter(logp[mask],d["HCO3_mmol"][mask],
                   facecolors="none",edgecolors=c,marker=m,s=MS,zorder=4,linewidths=1.3)
    if SHOW_LABELS:
        for xi,yi,n in zip(logp, d["Ca_mmol"], d["Name"]):
            if pd.isna(xi) or pd.isna(yi): continue
            ax.annotate(str(n),(xi,yi),xytext=(5,5),textcoords="offset points",
                        fontsize=6.5,color="#222",fontweight="500",zorder=6,clip_on=True,
                        bbox=dict(boxstyle="round,pad=0.18",fc="white",ec="none",alpha=0.7))
    lh = [
        mlines.Line2D([],[],color="black",ls="-", lw=1.5,marker="o",ms=4,label="Ca²⁺ open 15°C (PHREEQC)"),
        mlines.Line2D([],[],color="black",ls="--",lw=1.5,marker="s",ms=4,label="HCO₃⁻ open 15°C (PHREEQC)"),
        mlines.Line2D([],[],color="#888",marker="o",ls="None",ms=5,label="filled = Ca²⁺"),
        mlines.Line2D([],[],color="#888",marker="o",ls="None",ms=5,
                      markerfacecolor="none",markeredgewidth=1.3,label="open = HCO₃⁻"),
    ]
    if SYMBOLOGY == "AQUIFER":
        for a in sorted(d["Aquifer"].unique()):
            c,m=aquifer_style(a); lh.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=AQUIFER_PALETTE.get(a,(None,a))[1]))
    else:
        for k in sorted(d["Cluster"].unique()):
            c,m=cstyle(k); lh.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=f"Cluster {int(k)}"))
    ax.legend(handles=lh,fontsize=7.5,frameon=True,framealpha=0.92,edgecolor="#ddd",ncol=2)
    style(ax,"log pCO₂  (atm)","Concentration  (mmol/L)","log pCO₂ vs HCO₃⁻ and Ca²⁺")
    fig.tight_layout(); save(fig,"13_PCO2_vs_HCO3_Ca",out)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 14 — log pCO2 vs SI Calcite
# ══════════════════════════════════════════════════════════════════════════════
def plot_si_pco2(df, COL, out):
    pc  = COL.get("PCO2")
    sic = COL.get("siCal")
    if not pc or not sic or pc not in df.columns or sic not in df.columns:
        print("  SI Calcite vs pCO2 skipped (columns missing)"); return
    d = df.dropna(subset=[pc, sic]).copy()
    if d.empty: print("  SI Calcite vs pCO2 skipped (no data)"); return
    x = pd.to_numeric(d[pc],  errors="coerce")
    y = pd.to_numeric(d[sic], errors="coerce")
    fig,ax=plt.subplots(figsize=(6,5))
    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    ax.axhline(0, color="#555", lw=1.0, ls="--", zorder=1)
    handles_extra = [mlines.Line2D([],[],color="#555",ls="--",lw=1,label="SI = 0  (equilibrium)")]
    if SYMBOLOGY == "AQUIFER":
        for a in sorted(d["Aquifer"].unique()):
            c,m=aquifer_style(a); handles_extra.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=AQUIFER_PALETTE.get(a,(None,a))[1]))
    else:
        for k in sorted(d["Cluster"].unique()):
            c,m=cstyle(k); handles_extra.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=f"Cluster {int(k)}"))
    ax.legend(handles=handles_extra,fontsize=8,frameon=True,framealpha=0.92,edgecolor="#ddd")
    style(ax,"log pCO₂  (atm)","SI Calcite","log pCO₂ vs SI Calcite")
    fig.tight_layout(); save(fig,"14_PCO2_vs_SI_Calcite",out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 15 — log pCO₂ vs pH
# ══════════════════════════════════════════════════════════════════════════════
def plot_pco2_vs_ph(df, COL, out):
    pc_col = COL.get("PCO2")
    ph_col = COL.get("pH")
    if not pc_col or not ph_col or pc_col not in df.columns or ph_col not in df.columns:
        print("  log pCO₂ vs pH skipped (columns missing)"); return
    d = df.dropna(subset=[pc_col, ph_col]).copy()
    if d.empty: print("  log pCO₂ vs pH skipped (no data)"); return

    x = pd.to_numeric(d[pc_col],  errors="coerce")
    y = pd.to_numeric(d[ph_col],  errors="coerce")

    fig, ax = plt.subplots(figsize=(6, 5))

    # reference lines
    ax.axvline(-2.5, color="#aaa", lw=0.8, ls=":", zorder=1)
    ax.axvline(-3.5, color="#aaa", lw=0.8, ls=":", zorder=1)
    ax.text(-2.5, y.max(), " soil CO₂", fontsize=7.5, color="#888",
            va="top", ha="left")
    ax.text(-3.5, y.max(), " atmosphere", fontsize=7.5, color="#888",
            va="top", ha="left")

    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    if SYMBOLOGY == "AQUIFER":
        aq_legend(ax, d["Aquifer"])
    else:
        cl_legend(ax, d["Cluster"])
    style(ax, "log pCO₂  (atm)", "pH", "log pCO₂ vs pH")
    fig.tight_layout()
    save(fig, "15_PCO2_vs_pH", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 16 — δ¹³C vs ¹⁴C - Diagnostic areas
# ══════════════════════════════════════════════════════════════════════════════
def plot_14c_13c_areas(df, COL, out):
    c14_col = COL.get("C14")
    c13_col = COL.get("C13")
    if not c14_col or not c13_col:
        print("  14C/13C areas plot skipped (columns not found)"); return
    d = df.dropna(subset=[c14_col, c13_col]).copy()
    if d.empty:
        print("  14C/13C areas plot skipped (no data)"); return

    x = pd.to_numeric(d[c13_col], errors="coerce")
    y = pd.to_numeric(d[c14_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # ── Pearson area ──────────────────────────────────────────────────────────
    # Bounded by:
    #   left   : x = -25
    #   right  : x = -15
    #   bottom : y = 0
    #   top    : line from (-25, 100) to (-11, 50), clipped at x = -15
    slope = (50 - 100) / (-11 - (-25))          # -25/7
    y_at_x15 = 100 + slope * (-15 - (-25))      # line value at x=-15

    pearson_poly = np.array([
        [-25,    0        ],
        [-15,    0        ],
        [-15,    y_at_x15 ],
        [-25,    100      ],
    ])
    from matplotlib.patches import Polygon as MPoly
    ax.add_patch(MPoly(pearson_poly, closed=True,
                       facecolor="#4A90D9", alpha=0.18,
                       edgecolor="#4A90D9", linewidth=1.2,
                       zorder=1))
    ax.text(-20, 25, "Pearson\nmodel", ha="center", va="center",
            fontsize=8, color="#1a5fa8", fontstyle="italic",
            fontweight="600", zorder=2)

    # ── Eichinger area ────────────────────────────────────────────────────────
    # Bounded by:
    #   left   : x = -11  (vertical)
    #   bottom : y = 0    (x axis)
    #   top    : line from (-11, 50) to (0, 0)
    #
    # Line from (-11, 50) to (0, 0):
    #   slope = (0 - 50) / (0 - (-11)) = -50/11
    #   passes through (0,0): y = (-50/11) * x  → note x is negative so y>0
    eich_poly = np.array([
        [-11,  0 ],
        [  0,  0 ],
        [-11, 50 ],
    ])
    ax.add_patch(MPoly(eich_poly, closed=True,
                       facecolor="#E8A838", alpha=0.18,
                       edgecolor="#E8A838", linewidth=1.2,
                       zorder=1))
    ax.text(-5.5, 12, "Eichinger\nmodel", ha="center", va="center",
            fontsize=8, color="#a06010", fontstyle="italic",
            fontweight="600", zorder=2)

    # ── Mook area ─────────────────────────────────────────────────────────────
    # Bounded by:
    #   left   : x = -15
    #   right  : x = -11
    #   bottom : y = 0
    #   top    : line from (-15, 100) to (-11, 50)
    slope_mook = (50 - 100) / (-11 - (-15))     # -50/4 = -12.5
    y_mook_left  = 100                            # at x=-15
    y_mook_right = 50                             # at x=-11

    mook_poly = np.array([
        [-15,   0            ],
        [-11,   0            ],
        [-11,   y_mook_right ],
        [-15,   y_mook_left  ],
    ])
    ax.add_patch(MPoly(mook_poly, closed=True,
                       facecolor="#E87040", alpha=0.18,
                       edgecolor="#E87040", linewidth=1.2,
                       zorder=1))
    ax.text(-13, 25, "Mook\nmodel", ha="center", va="center",
            fontsize=8, color="#b84a10", fontstyle="italic",
            fontweight="600", zorder=2)

    # ── data points ───────────────────────────────────────────────────────────
    c14s = COL.get("C14s"); c13s = COL.get("C13s")
    _sym_groups = sorted(d["Aquifer"].unique()) if SYMBOLOGY == "AQUIFER" else sorted(d["Cluster"].unique())
    for grp in _sym_groups:
        mask = (d["Aquifer"]==grp) if SYMBOLOGY == "AQUIFER" else (d["Cluster"]==grp)
        c, m = (aquifer_style(grp) if SYMBOLOGY == "AQUIFER" else cstyle(grp))
        if c14s and c14s in df.columns and c13s and c13s in df.columns:
            ax.errorbar(x[mask], y[mask],
                        xerr=pd.to_numeric(d[c13s][mask], errors="coerce"),
                        yerr=pd.to_numeric(d[c14s][mask], errors="coerce"),
                        fmt="none", color=c, alpha=0.45, lw=0.9,
                        capsize=2, zorder=3)
        ax.scatter(x[mask], y[mask], color=c, marker=m,
                   s=MS, zorder=4, edgecolors="white", linewidths=0.4)

    if SHOW_LABELS:
        for xi, yi, n in zip(x, y, d["Name"]):
            if pd.isna(xi) or pd.isna(yi): continue
            ax.annotate(str(n), (xi, yi), xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=6.5, color="#222", fontweight="500",
                        zorder=6, clip_on=True,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec="none", alpha=0.7))

    if SYMBOLOGY == "AQUIFER":
        aq_legend(ax, d["Aquifer"])
    else:
        cl_legend(ax, d["Cluster"])
    style(ax, "δ¹³C  (‰ VPDB)", "¹⁴C  (pmc)",
          "δ¹³C vs ¹⁴C — Diagnostic areas")
    ax.set_xlim(-30, 5)
    ax.set_ylim(0, 110)
    fig.tight_layout()
    save(fig, "16_14C_13C_areas", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 17 — δ¹³C vs ¹⁴C — Mixing lines
# ══════════════════════════════════════════════════════════════════════════════
def plot_14c_13c_lines(df, COL, out):
    and_col = COL.get("C14")
    c13_col = COL.get("C13")
    if not and_col or not c13_col or and_col not in df.columns or c13_col not in df.columns:
        print("  C14 vs C13 (lines) skipped (columns missing)"); return
    d = df.dropna(subset=[and_col, c13_col]).copy()
    if d.empty: print("  C14 vs C13 (lines) skipped (no data)"); return
    x = pd.to_numeric(d[c13_col], errors="coerce")
    y = pd.to_numeric(d[and_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # mixing line: (-25, 100) → (0, 0)
    ax.plot([-25, 0], [100, 0], color="#185FA5", lw=1.5, ls="-", zorder=1)
    ax.text(-25, 103, "soil CO₂ C3 cover\n(δ¹³C = −25‰)",
            ha="left", va="bottom", fontsize=7.5, color="#185FA5")

    # mixing line: (-15, 100) → (0, 0)
    ax.plot([-15, 0], [100, 0], color="#D85A30", lw=1.5, ls="-", zorder=1)
    ax.text(-15, 103, "soil CO₂ C4 cover\n(δ¹³C = −15‰)",
            ha="left", va="bottom", fontsize=7.5, color="#D85A30")

    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    if SYMBOLOGY == "AQUIFER":
        aq_legend(ax, d["Aquifer"])
    else:
        cl_legend(ax, d["Cluster"])
    style(ax, "δ¹³C  (‰ VPDB)",
              "¹⁴C  (pmc)",
              "δ¹³C vs ¹⁴C — Mixing lines")
    ax.set_xlim(-30, 5)
    ax.set_ylim(0, 110)
    fig.tight_layout()
    save(fig, "17_14C_vs_d13C_lines", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 18 —  δ¹³C vs ¹⁴C — C3 Pearson Mixing line
# ══════════════════════════════════════════════════════════════════════════════
def plot_14c_13c_pearson(df, COL, out):
    c14_col = COL.get("C14")
    c13_col = COL.get("C13")
    if not c14_col or not c13_col:
        print("  14C/13C Pearson plot skipped (columns not found)"); return
    d = df.dropna(subset=[c14_col, c13_col]).copy()
    if d.empty:
        print("  14C/13C Pearson plot skipped (no data)"); return

    x = pd.to_numeric(d[c13_col], errors="coerce")
    y = pd.to_numeric(d[c14_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Pearson mixing line: (-25, 100) → (0, 0)
    ax.plot([-25, 0], [100, 0], color="black", lw=1.5, ls="-", zorder=1,
            label="Pearson mixing line")
    # endmember points
    ax.scatter([-25], [100], color="#000000", s=80, zorder=3,
               edgecolors="white", lw=1)
    ax.scatter([0],   [0],   color="#000000", s=80, zorder=3,
               edgecolors="white", lw=1)
    ax.text(-26.5, 101.5, "Soil CO₂", fontsize=8, color="#000000",
            fontweight="600", va="bottom")
    ax.text(0.3,   1.5,   "Carbonate", fontsize=8, color="#000000",
            fontweight="600", va="bottom")

    # data points with error bars if available
    c14s = COL.get("C14s"); c13s = COL.get("C13s")
    _sym_groups = sorted(d["Aquifer"].unique()) if SYMBOLOGY == "AQUIFER" else sorted(d["Cluster"].unique())
    for grp in _sym_groups:
        mask = (d["Aquifer"]==grp) if SYMBOLOGY == "AQUIFER" else (d["Cluster"]==grp)
        c, m = (aquifer_style(grp) if SYMBOLOGY == "AQUIFER" else cstyle(grp))
        if c14s and c14s in df.columns and c13s and c13s in df.columns:
            ax.errorbar(x[mask], y[mask],
                        xerr=pd.to_numeric(d[c13s][mask], errors="coerce"),
                        yerr=pd.to_numeric(d[c14s][mask], errors="coerce"),
                        fmt="none", color=c, alpha=0.45, lw=0.9,
                        capsize=2, zorder=2)
        ax.scatter(x[mask], y[mask], color=c, marker=m,
                   s=MS, zorder=4, edgecolors="white", linewidths=0.4)

    if SHOW_LABELS:
        for xi, yi, n in zip(x, y, d["Name"]):
            if pd.isna(xi) or pd.isna(yi): continue
            ax.annotate(str(n), (xi, yi), xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=6.5, color="#222", fontweight="500",
                        zorder=6, clip_on=True,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec="none", alpha=0.7))

    if SYMBOLOGY == "AQUIFER":
        aq_legend(ax, d["Aquifer"])
    else:
        cl_legend(ax, d["Cluster"])
    style(ax, "δ¹³C  (‰ VPDB)", "¹⁴C  (pmc)", "¹⁴C vs δ¹³C — Pearson mixing line")
    ax.set_xlim(-30, 5)
    ax.set_ylim(0, 110)
    fig.tight_layout()
    save(fig, "18_14C_13C_pearson", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 19 — δ¹³C observed vs δ¹³C computed NETPATH
# ══════════════════════════════════════════════════════════════════════════════
def plot_d13c_comp_vs_obs(df, COL, out):
    c13c_col = COL.get("C13c")
    c13_col  = COL.get("C13")
    if not c13c_col or not c13_col or c13c_col not in df.columns or c13_col not in df.columns:
        print("  d13C computed vs observed skipped (columns missing)"); return
    d = df.dropna(subset=[c13c_col, c13_col]).copy()
    if d.empty: print("  d13C computed vs observed skipped (no data)"); return
    x = pd.to_numeric(d[c13_col],  errors="coerce")   # observed  → x axis
    y = pd.to_numeric(d[c13c_col], errors="coerce")   # computed  → y axis
    fig, ax = plt.subplots(figsize=(6, 5))
    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    # 1:1 line across the data range
    lo = min(x.min(), y.min()) - 1
    hi = max(x.max(), y.max()) + 1
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.1, ls="--", zorder=1)
    handles_extra = [mlines.Line2D([],[],color="black",ls="--",lw=1.1,label="1:1")]
    if SYMBOLOGY == "AQUIFER":
        for a in sorted(d["Aquifer"].unique()):
            c, m = aquifer_style(a)
            handles_extra.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=AQUIFER_PALETTE.get(a,(None,a))[1]))
    else:
        for k in sorted(d["Cluster"].unique()):
            c, m = cstyle(k)
            handles_extra.append(mlines.Line2D([],[],color=c,marker=m,
                                 ls="None",ms=6,label=f"Cluster {int(k)}"))
    ax.legend(handles=handles_extra, fontsize=8, frameon=True,
              framealpha=0.92, edgecolor="#ddd")
    style(ax, "δ¹³C observed  (‰ VPDB)",
              "δ¹³C computed  (‰ VPDB)",
              "δ¹³C observed vs δ¹³C computed NETPATH")
    fig.tight_layout()
    save(fig, "19_d13C_computed_vs_obs", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 20 — ¹⁴C Aobs vs ¹⁴C Acorr NETPATH
# ══════════════════════════════════════════════════════════════════════════════
def plot_acorr_vs_14_netpath(df, COL, out):
    acorr_col = COL.get("Acorr")
    c14_col   = COL.get("C14")
    if not acorr_col or not c14_col or acorr_col not in df.columns or c14_col not in df.columns:
        print("  ¹⁴C Aobs vs ¹⁴C Acorr NETPATH (columns missing)"); return
    d = df.dropna(subset=[acorr_col, c14_col]).copy()
    if d.empty: print("  ¹⁴C Aobs vs ¹⁴C Acorr NETPATH skipped (no data)"); return
    x = pd.to_numeric(d[c14_col],   errors="coerce")   # measured ¹⁴C  → x axis
    y = pd.to_numeric(d[acorr_col], errors="coerce")   # corrected (decay only) → y axis
    fig, ax = plt.subplots(figsize=(6, 5))
    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    # 1:1 line
    hi = max(x.max(), y.max()) * 1.08
    ax.plot([0, hi], [0, hi], color="black", lw=1.1, ls="--", zorder=1)
    handles_extra = [mlines.Line2D([],[],color="black",ls="--",lw=1.1,label="1:1")]
    if SYMBOLOGY == "AQUIFER":
        for a in sorted(d["Aquifer"].unique()):
            c, m = aquifer_style(a)
            handles_extra.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=AQUIFER_PALETTE.get(a,(None,a))[1]))
    else:
        for k in sorted(d["Cluster"].unique()):
            c, m = cstyle(k)
            handles_extra.append(mlines.Line2D([],[],color=c,marker=m,
                                 ls="None",ms=6,label=f"Cluster {int(k)}"))
    ax.legend(handles=handles_extra, fontsize=8, frameon=True,
              framealpha=0.92, edgecolor="#ddd")
    style(ax, "¹⁴C observed  (pmc)",
              "¹⁴C corrected  (pmc)  — NETPATH",
              "¹⁴C Aobs vs Acorr NETPATH")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "20_AcorrNETPATH_vs_14C", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 21 — δ¹³C vs ¹⁴C And NETPATH
# ══════════════════════════════════════════════════════════════════════════════
def plot_and_d13c_netpath(df, COL, out):
    and_col = COL.get("And")
    c13_col = COL.get("C13")
    if not and_col or not c13_col or and_col not in df.columns or c13_col not in df.columns:
        print("  δ¹³C vs ¹⁴C And NETPATH skipped (columns missing)"); return
    d = df.dropna(subset=[and_col, c13_col]).copy()
    if d.empty: print("  δ¹³C vs ¹⁴C And NETPATH skipped (no data)"); return
    x = pd.to_numeric(d[c13_col], errors="coerce")
    y = pd.to_numeric(d[and_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # mixing line: (-25, 100) → (0, 0)
    ax.plot([-25, 0], [100, 0], color="#185FA5", lw=1.5, ls="-", zorder=1)
    ax.text(-25, 103, "soil CO₂ C3 cover\n(δ¹³C = −25‰)",
            ha="left", va="bottom", fontsize=7.5, color="#185FA5")

    # mixing line: (-15, 100) → (0, 0)
    ax.plot([-15, 0], [100, 0], color="#D85A30", lw=1.5, ls="-", zorder=1)
    ax.text(-15, 103, "soil CO₂ C4 cover\n(δ¹³C = −15‰)",
            ha="left", va="bottom", fontsize=7.5, color="#D85A30")

    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    if SYMBOLOGY == "AQUIFER":
        aq_legend(ax, d["Aquifer"])
    else:
        cl_legend(ax, d["Cluster"])
    style(ax, "δ¹³C  (‰ VPDB)",
              "¹⁴C  (pmc)  — no decay",
              "δ¹³C vs ¹⁴C And NETPATH")
    ax.set_xlim(-30, 5)
    ax.set_ylim(0, 110)
    fig.tight_layout()
    save(fig, "21_And_NETPATH_vs_d13C", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 22 — δ¹³C vs ¹⁴C Acorr NETPATH
# ══════════════════════════════════════════════════════════════════════════════
def plot_acorr_d13c_lines(df, COL, out):
    and_col = COL.get("Acorr")
    c13_col = COL.get("C13")
    if not and_col or not c13_col or and_col not in df.columns or c13_col not in df.columns:
        print("  δ¹³C vs ¹⁴C Acorr NETPATH skipped (columns missing)"); return
    d = df.dropna(subset=[and_col, c13_col]).copy()
    if d.empty: print("  δ¹³C vs ¹⁴C Acorr NETPATH skipped (no data)"); return
    x = pd.to_numeric(d[c13_col], errors="coerce")
    y = pd.to_numeric(d[and_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # mixing line: (-25, 100) → (0, 0)
    ax.plot([-25, 0], [100, 0], color="#185FA5", lw=1.5, ls="-", zorder=1)
    ax.text(-25, 103, "soil CO₂ C3 cover\n(δ¹³C = −25‰)",
            ha="left", va="bottom", fontsize=7.5, color="#185FA5")

    # mixing line: (-15, 100) → (0, 0)
    ax.plot([-15, 0], [100, 0], color="#D85A30", lw=1.5, ls="-", zorder=1)
    ax.text(-15, 103, "soil CO₂ C4 cover\n(δ¹³C = −15‰)",
            ha="left", va="bottom", fontsize=7.5, color="#D85A30")

    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    if SYMBOLOGY == "AQUIFER":
        aq_legend(ax, d["Aquifer"])
    else:
        cl_legend(ax, d["Cluster"])
    style(ax, "δ¹³C  (‰ VPDB)",
              "¹⁴C  (pmc)  — Corrected NETPATH",
              "δ¹³C vs ¹⁴C Acorr NETPATH")
    ax.set_xlim(-30, 5)
    ax.set_ylim(0, 110)
    fig.tight_layout()
    save(fig, "22_d13C_vs_Acorr_NETPATH", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 23 — ¹⁴C Acorr - NETPATH vs Pearson
# ══════════════════════════════════════════════════════════════════════════════
def plot_acorr_comparison(df, COL, out):
    np_col = COL.get("Acorr")
    pr_col = COL.get("AcorrP")
    if not np_col or not pr_col or np_col not in df.columns or pr_col not in df.columns:
        print("  ¹⁴C Acorr - NETPATH vs Pearson skipped (columns missing)"); return
    d = df.dropna(subset=[np_col, pr_col]).copy()
    if d.empty: print("  ¹⁴C Acorr - NETPATH vs Pearson skipped (no data)"); return
    x = pd.to_numeric(d[np_col], errors="coerce")
    y = pd.to_numeric(d[pr_col], errors="coerce")
    fig, ax = plt.subplots(figsize=(6, 5))
    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    lo = min(x.min(), y.min()) * 0.95
    hi = max(x.max(), y.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.1, ls="--", zorder=1)
    handles_extra = [mlines.Line2D([],[],color="black",ls="--",lw=1.1,label="1:1")]
    if SYMBOLOGY == "AQUIFER":
        for a in sorted(d["Aquifer"].unique()):
            c, m = aquifer_style(a)
            handles_extra.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=AQUIFER_PALETTE.get(a,(None,a))[1]))
    else:
        for k in sorted(d["Cluster"].unique()):
            c, m = cstyle(k)
            handles_extra.append(mlines.Line2D([],[],color=c,marker=m,
                                 ls="None",ms=6,label=f"Cluster {int(k)}"))
    ax.legend(handles=handles_extra,fontsize=8,frameon=True,
              framealpha=0.92,edgecolor="#ddd")
    style(ax, "¹⁴C corrected — NETPATH  (pmc)",
              "¹⁴C corrected — Pearson  (pmc)",
              "Corrected ¹⁴C activity - NETPATH vs Pearson")
    fig.tight_layout()
    save(fig, "23_Acorr_NETPATH_vs_Pearson", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 24 — Radiocarbon age - NETPATH vs Pearson
# ══════════════════════════════════════════════════════════════════════════════
def plot_age_comparison(df, COL, out):
    np_col = COL.get("AgeNP")
    pr_col = COL.get("AgePR")
    if not np_col or not pr_col or np_col not in df.columns or pr_col not in df.columns:
        print("  Radiocarbon age - NETPATH vs Pearson skipped (columns missing)"); return
    d = df.dropna(subset=[np_col, pr_col]).copy()
    if d.empty: print("  Radiocarbon age - NETPATH vs Pearson skipped (no data)"); return
    x = pd.to_numeric(d[np_col], errors="coerce")
    y = pd.to_numeric(d[pr_col], errors="coerce")
    fig, ax = plt.subplots(figsize=(6, 5))
    scat(ax, x, y, d["Cluster"], names=d["Name"], aquifer=d["Aquifer"])
    lo = min(0, min(x.min(), y.min()) * 0.95)
    hi = max(x.max(), y.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.1, ls="--", zorder=1)
    handles_extra = [mlines.Line2D([],[],color="black",ls="--",lw=1.1,label="1:1")]
    if SYMBOLOGY == "AQUIFER":
        for a in sorted(d["Aquifer"].unique()):
            c, m = aquifer_style(a)
            handles_extra.append(mlines.Line2D([],[],color=c,marker=m,ls="None",ms=6,label=AQUIFER_PALETTE.get(a,(None,a))[1]))
    else:
        for k in sorted(d["Cluster"].unique()):
            c, m = cstyle(k)
            handles_extra.append(mlines.Line2D([],[],color=c,marker=m,
                                 ls="None",ms=6,label=f"Cluster {int(k)}"))
    ax.legend(handles=handles_extra,fontsize=8,frameon=True,
              framealpha=0.92,edgecolor="#ddd")
    style(ax, "Model age — NETPATH (years BP)",
              "Model age — Pearson (years BP)",
              "Model age - NETPATH vs Pearson")
    fig.tight_layout()
    save(fig, "24_Age_NETPATH_vs_Pearson", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 25 — Acorr - Pearson / NETPATH ratio
# ══════════════════════════════════════════════════════════════════════════════
def plot_ratio_bar(df, COL, out):
    np_col = COL.get("Acorr")
    pr_col = COL.get("AcorrP")
    bh_col    = COL.get("Borehole")
    if not np_col or not pr_col or np_col not in df.columns or pr_col not in df.columns:
        print("  Acorr - Pearson / NETPATH ratio skipped (columns missing)"); return
    d = df.dropna(subset=[np_col, pr_col]).copy()
    if d.empty: print("  Acorr - Pearson / NETPATH ratio skipped (no data)"); return

    # use Borehole column if found, otherwise Name
    if bh_col and bh_col in df.columns:
        d["_label"] = d[bh_col].astype(str)
    else:
        d["_label"] = d["Name"].astype(str)

    d["_ratio"] = pd.to_numeric(d[pr_col], errors="coerce") / pd.to_numeric(d[np_col], errors="coerce")
    d = d.dropna(subset=["_ratio"]).sort_values("_ratio")

    fig, ax = plt.subplots(figsize=(max(8, len(d)*0.55), 5))

    for i, (_, row) in enumerate(d.iterrows()):
        if SYMBOLOGY == "AQUIFER":
            aq = row["Aquifer"] if "Aquifer" in d.columns else "Unknown"
            c, _ = aquifer_style(aq)
        else:
            k   = int(row["Cluster"]) if "Cluster" in d.columns else 1
            c, _= cstyle(k)
        ax.bar(i, row["_ratio"], color=c, edgecolor="white",
               linewidth=0.5, zorder=2)

    ax.axhline(1.0, color="black", lw=1.1, ls="--", zorder=3,
               label="Ratio = 1  (models agree)")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["_label"].tolist(),
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pearson ¹⁴C corrected / NETPATH ¹⁴C corrected", fontsize=10)
    ax.set_title("Acorr - Pearson / NETPATH ratio",
                 fontsize=11, fontweight="600", pad=8)
    ax.spines[["top","right"]].set_visible(False)

    # symbology legend
    handles = [mlines.Line2D([],[],color="black",ls="--",lw=1.1,
                              label="Ratio = 1")]
    if SYMBOLOGY == "AQUIFER":
        for a in sorted(d["Aquifer"].unique() if "Aquifer" in d.columns else ["Unknown"]):
            c, _ = aquifer_style(a)
            handles.append(mpatches.Patch(facecolor=c, edgecolor="white",
                                          label=AQUIFER_PALETTE.get(a,(None,a))[1]))
    else:
        for k in sorted(d["Cluster"].unique() if "Cluster" in d.columns else [1]):
            c, m = cstyle(k)
            handles.append(mpatches.Patch(facecolor=c, edgecolor="white",
                                          label=f"Cluster {int(k)}"))
    ax.legend(handles=handles, fontsize=8, frameon=True,
              framealpha=0.92, edgecolor="#ddd")
    fig.tight_layout()
    save(fig, "25_Acorrected_ratio_Pearson_NETPATH", out)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 26 — Radiocarbon age - Pearson / NETPATH ratio
# ══════════════════════════════════════════════════════════════════════════════
def plot_age_ratio_bar(df, COL, out):
    np_col = COL.get("AgeNP")
    pr_col = COL.get("AgePR")
    bh_col = COL.get("Borehole")
    if not np_col or not pr_col or np_col not in df.columns or pr_col not in df.columns:
        print("  Radiocarbon age - Pearson / NETPATH ratio skipped (columns missing)"); return
    d = df.dropna(subset=[np_col, pr_col]).copy()
    d["_np"]  = pd.to_numeric(d[np_col], errors="coerce")
    d["_pr"]  = pd.to_numeric(d[pr_col], errors="coerce")
    d = d.dropna(subset=["_np","_pr"])
    # avoid division by zero
    d = d[d["_np"] != 0]
    if d.empty: print("  Radiocarbon age - Pearson / NETPATH ratio skipped (no data)"); return
    d["_ratio"] = d["_pr"] / d["_np"]
    d = d.sort_values("_ratio")
 
    if bh_col and bh_col in df.columns:
        d["_label"] = d[bh_col].astype(str)
    else:
        d["_label"] = d["Name"].astype(str)
 
    fig, ax = plt.subplots(figsize=(max(8, len(d)*0.55), 5))
 
    for i, (_, row) in enumerate(d.iterrows()):
        if SYMBOLOGY == "AQUIFER":
            aq = row["Aquifer"] if "Aquifer" in d.columns else "Unknown"
            c, _ = aquifer_style(aq)
        else:
            k    = int(row["Cluster"]) if "Cluster" in d.columns else 1
            c, _ = cstyle(k)
        ax.bar(i, row["_ratio"], color=c, edgecolor="white",
               linewidth=0.5, zorder=2)
 
    ax.axhline(1.0, color="black", lw=1.1, ls="--", zorder=3)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["_label"].tolist(),
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pearson age / NETPATH age", fontsize=10)
    ax.set_title("Radiocarbon age - Pearson / NETPATH ratio",
                 fontsize=11, fontweight="600", pad=8)
    ax.spines[["top","right"]].set_visible(False)
 
    handles = [mlines.Line2D([],[],color="black",ls="--",lw=1.1,
                              label="Ratio = 1  (models agree)")]
    if SYMBOLOGY == "AQUIFER":
        for a in sorted(d["Aquifer"].unique() if "Aquifer" in d.columns else ["Unknown"]):
            c, _ = aquifer_style(a)
            handles.append(mpatches.Patch(facecolor=c, edgecolor="white",
                                          label=AQUIFER_PALETTE.get(a,(None,a))[1]))
    else:
        for k in sorted(d["Cluster"].unique() if "Cluster" in d.columns else [1]):
            c, _ = cstyle(k)
            handles.append(mpatches.Patch(facecolor=c, edgecolor="white",
                                          label=f"Cluster {int(k)}"))
    ax.legend(handles=handles, fontsize=8, frameon=True,
              framealpha=0.92, edgecolor="#ddd")
    fig.tight_layout()
    save(fig, "26_Radiocarbon_age_Pearson_NETPATH", out)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if len(sys.argv)<2: print(__doc__); sys.exit(0)
    # Parse optional --symbology and --label-column flags before the path argument
    global SYMBOLOGY, LABEL_COLUMN, FILTER, FILTER_COLUMN, FILTER_VALUE
    args = sys.argv[1:]
    if "--symbology" in args:
        idx = args.index("--symbology")
        if idx + 1 < len(args):
            val = args[idx + 1].upper()
            if val in ("CLUSTER", "AQUIFER"):
                SYMBOLOGY = val
                args = args[:idx] + args[idx+2:]
            else:
                print(f"  WARNING: unknown --symbology value '{args[idx+1]}', using '{SYMBOLOGY}'")
                args = args[:idx] + args[idx+2:]
    if "--label-column" in args:
        idx = args.index("--label-column")
        if idx + 1 < len(args):
            LABEL_COLUMN = args[idx + 1]
            args = args[:idx] + args[idx+2:]
        else:
            print("  WARNING: --label-column requires a value, ignoring")
    if "--filter-column" in args:
        idx = args.index("--filter-column")
        if idx + 1 < len(args):
            FILTER_COLUMN = args[idx + 1]
            FILTER = True
            args = args[:idx] + args[idx+2:]
        else:
            print("  WARNING: --filter-column requires a value, ignoring")
    if "--filter-value" in args:
        idx = args.index("--filter-value")
        # Collect all subsequent non-flag tokens as filter values
        vals = []
        j = idx + 1
        while j < len(args) and not args[j].startswith("--"):
            vals.append(args[j])
            j += 1
        if vals:
            FILTER_VALUE = vals if len(vals) > 1 else vals[0]
            FILTER = True
            args = args[:idx] + args[j:]
        else:
            print("  WARNING: --filter-value requires at least one value, ignoring")
    path = args[0]
    out  = os.path.dirname(os.path.abspath(path))
    print(f"\nLoading: {path}")
    print(f"  Symbology mode: {SYMBOLOGY}")
    if SHOW_LABELS:
        print(f"  Labels: ON  (column: {'auto-detect' if LABEL_COLUMN is None else LABEL_COLUMN})")
    df,COL=load(path)
    df = apply_filter(df)
    print(f"  {len(df)} samples  |  clusters: {sorted(df['Cluster'].unique())}")
    print(f"  matched columns: {[k for k,v in COL.items() if v]}")
    print("\nGenerating figures:")
    plot_cai(df,out)
    plot_na_norm(df,out)
    plot_nacl(df,out)
    plot_cacl(df,out)
    plot_pco2(df,COL,out)
    plot_hco3_camg(df,out)
    plot_si_pco2(df,COL,out)
    plot_ec_tds(df,COL,out)
    plot_exchange_balance(df,out)
    plot_14c_13c_areas(df,COL,out)
    plot_14c_13c_pearson(df,COL,out)
    plot_pco2_vs_ph(df,COL,out)
    plot_hco3_vs_ph(df,COL,out)
    plot_and_d13c_netpath(df,COL,out)
    plot_d13c_comp_vs_obs(df,COL,out)
    plot_acorr_vs_14_netpath(df,COL,out)
    plot_14c_13c_lines(df,COL,out)
    plot_acorr_d13c_lines(df,COL,out)
    plot_acorr_comparison(df,COL,out)
    plot_age_comparison(df,COL,out)
    plot_ratio_bar(df,COL,out)
    plot_age_ratio_bar(df,COL,out)
    plot_na_vs_cl(df,out)
    plot_ca_vs_cl(df,out)
    plot_so4_vs_cl(df,out)
    plot_ca_vs_so4(df,out)
    print("\nDone.")
 
if __name__=="__main__":
    main()