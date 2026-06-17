"""
hydrochem_dashboard.py
──────────────────────
Interactive Streamlit dashboard for hydrochemical data visualisation.
Map panel (Lambert 93 → WGS84) + 26 Plotly plots with cross-highlighting.

Run with:
    streamlit run hydrochem_dashboard.py
"""

import warnings, io
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyproj import Transformer
import streamlit as st
import tempfile, os, zipfile

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
PALETTE        = ["#185FA5","#D85A30","#1D9E75","#BA7517","#993556","#534AB7","#3B6D11"]
PLOTLY_SYMBOLS = ["circle","triangle-up","square","diamond","triangle-down","cross","x"]

AQUIFER_PALETTE = {
    "PlioQ":     "#555555",
    "Miocene":   "#E0CA00",
    "Eocene":    "#E86000",
    "Oligocene": "#6A0DAD",
}
_AQ_ALIASES = {
    "PlioQ":     {"plioq","plioquaternary","plioquaternay","plio-quaternary",
                  "plio quaternary","plioquet","plioqu"},
    "Miocene":   {"miocene","mio"},
    "Eocene":    {"eocene","eoc"},
    "Oligocene": {"oligocene","oligocen","olig","oli"},
}
_AQ_LOOKUP = {a: c for c, aliases in _AQ_ALIASES.items() for a in aliases}

MW  = dict(Na=22.990,K=39.098,Mg=24.305,Ca=40.078,
           Cl=35.453,HCO3=61.016,SO4=96.06,NH4=18.038,F=18.998,NO3=62.004)
VAL = dict(Na=1,K=1,Mg=2,Ca=2,Cl=1,HCO3=1,SO4=2,NH4=1,F=1,NO3=1)
meq  = lambda s,i: pd.to_numeric(s,errors="coerce")*VAL[i]/MW[i]
mmol = lambda s,i: pd.to_numeric(s,errors="coerce")/MW[i]

LOOKUPS = {
    "Na":["na (ppm)","na"],"K":["k (ppm)","k"],"Mg":["mg (ppm)","mg"],
    "Ca":["ca (ppm)","ca"],"Cl":["cl (ppm)","cl"],"HCO3":["hco3 (ppm)","hco3"],
    "SO4":["so4 (ppm)","so4"],"NH4":["nh4 (ppm)","nh4"],"F":["f (ppm)","f"],
    "NO3":["no3 (ppm)","no3"],"pH":["ph"],"T":["t","t (°c)","t (oc)","temp","temperature"],
    "CE":["ce","ce (µs/cm)","ce (us/cm)","ec","conductivity"],
    "DO":["do","do2","do2 (%)","do (%)","dissolved oxygen"],
    "C14":["14c (pmc)","14c","c14"],"C14s":["14c stdev","14c std"],
    "C13":["d13c","δ13c"],"C13s":["d13c stdev","d13c std"],
    "D2H":["d2h","δ2h","d2h (‰)"],"D2Hs":["d2h stdev","d2h std"],
    "D18O":["d18o","δ18o","d18o (‰)"],"D18Os":["d18o stdev","d18o std"],
    "PCO2":["p_co2(g)","p_co2","pco2","log pco2"],
    "siCal":["si_calcite","si calcite"],"siDol":["si_dolomite","si dolomite"],
    "siGyp":["si_gypsum","si gypsum"],"siF":["si_fluorite","si fluorite"],
    "SAR":["sar"],"TDS":["tds","total dissolved solids","tds (mg/l)","tds (ppm)",
                          "mineralization","mineralisation"],
    "Li":["li (ppm)","li"],"Br":["br (ppm)","br"],"PO4":["po4 (ppm)","po4"],
    "And":["And_Netpath","And Netpath","and_netpath","and netpath"],
    "C13c":["13c_computed","d13c_computed","13c computed","d13c computed"],
    "Acorr":["Acorr_Netpath","Acorr Netpath","acorr_netpath","acorr netpath"],
    "AndP":["And_Pearson","And Pearson","and_pearson","and pearson"],
    "AcorrP":["Acorr_Pearson","Acorr Pearson","acorr_pearson","acorr pearson"],
    "AgeNP":["Age_Netpath","Age Netpath","age_netpath","age netpath"],
    "AgePR":["Age_Pearson","Age Pearson","age_pearson","age pearson"],
    "RatioPNP":["ratio_pearson_netpath","ratio pearson netpath"],
    "Borehole":["borehole","well","well name","bore"],
}

# Lambert 93 → WGS84 transformer (created once)
_L93_TO_WGS84 = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def normalize_aquifer(val):
    if pd.isna(val): return "Unknown"
    return _AQ_LOOKUP.get(str(val).strip().lower(), str(val).strip())

def fcol(df, key):
    low = {c.lower().strip(): c for c in df.columns}
    for cand in LOOKUPS.get(key, [key.lower()]):
        if cand in low: return low[cand]
    return None

def color_for(group, symbology, all_groups):
    if symbology == "AQUIFER":
        return AQUIFER_PALETTE.get(group, "#888888")
    idx = list(all_groups).index(group) if group in all_groups else 0
    return PALETTE[idx % len(PALETTE)]

def symbol_for(group, symbology, all_groups):
    idx = list(all_groups).index(group) if group in all_groups else 0
    return PLOTLY_SYMBOLS[idx % len(PLOTLY_SYMBOLS)]

def get_groups(d, symbology):
    col = d["Aquifer"] if symbology == "AQUIFER" else d["Cluster"]
    return sorted(col.dropna().unique())

def group_col(d, symbology):
    return d["Aquifer"] if symbology == "AQUIFER" else d["Cluster"]

def group_label(g, symbology):
    return str(g) if symbology == "AQUIFER" else f"Cluster {int(g)}"

def detect_coord_cols(df):
    """Return (x_col, y_col) for Lambert 93, or (None, None)."""
    low = {c.lower().strip(): c for c in df.columns}
    x_candidates = ["x l93 m","x l93","xl93","x_l93","x (l93)","x","xl93m"]
    y_candidates = ["y l93 m","y l93","yl93","y_l93","y (l93)","y","yl93m"]
    xc = next((low[k] for k in x_candidates if k in low), None)
    yc = next((low[k] for k in y_candidates if k in low), None)
    return xc, yc

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data(file_bytes, filename):
    import chardet
    enc = chardet.detect(file_bytes[:20000])["encoding"] or "latin-1"
    text = file_bytes.decode(enc, errors="replace")
    sep  = ";" if text.split("\n")[0].count(";") > text.split("\n")[0].count(",") else ","
    df   = pd.read_csv(io.StringIO(text), sep=sep)
    df.columns = df.columns.str.strip()

    # Name column
    name_candidates = ["name","sample","sample name","sample id","sampleid","id","code","borehole","well"]
    low = {c.lower().strip(): c for c in df.columns}
    name_col = next((low[n] for n in name_candidates if n in low), df.columns[0])
    df["Name"] = df[name_col].astype(str)

    # Cluster & Aquifer
    cc = next((c for c in df.columns if c.lower() == "cluster"), None)
    df["Cluster"] = pd.to_numeric(df[cc], errors="coerce").fillna(1).astype(int) if cc else 1
    aq_raw = next((c for c in df.columns if c.strip().lower() == "aquifer"), None)
    df["Aquifer"] = df[aq_raw].apply(normalize_aquifer) if aq_raw else "Unknown"

    # Derived ion columns
    COL = {k: fcol(df, k) for k in LOOKUPS}
    for ion in ["Na","K","Mg","Ca","Cl","HCO3","SO4","NH4","F","NO3"]:
        if COL[ion]:
            df[f"{ion}_meq"]  = meq(df[COL[ion]], ion)
            df[f"{ion}_mmol"] = mmol(df[COL[ion]], ion)

    # Lambert 93 → WGS84
    xc, yc = detect_coord_cols(df)
    if xc and yc:
        xs = pd.to_numeric(df[xc], errors="coerce")
        ys = pd.to_numeric(df[yc], errors="coerce")
        mask = xs.notna() & ys.notna()
        lons = np.full(len(df), np.nan)
        lats = np.full(len(df), np.nan)
        if mask.any():
            lons[mask], lats[mask] = _L93_TO_WGS84.transform(xs[mask].values, ys[mask].values)
        df["_lon"] = lons
        df["_lat"] = lats
    else:
        df["_lon"] = np.nan
        df["_lat"] = np.nan

    return df, COL

# ══════════════════════════════════════════════════════════════════════════════
#  SHAPEFILE LOADING
# ══════════════════════════════════════════════════════════════════════════════
# Shapefile palette — one colour per loaded layer
_SHP_COLORS = ["#E63946","#2196F3","#4CAF50","#FF9800","#9C27B0",
               "#00BCD4","#FF5722","#607D8B","#795548","#009688"]

@st.cache_data
def load_shapefile(file_dict):
    """
    Load a shapefile from a dict of {filename: bytes}.
    Reprojects to WGS84.  Returns a GeoDataFrame or None.
    """
    import geopandas as gpd

    with tempfile.TemporaryDirectory() as tmp:
        for fname, fbytes in file_dict.items():
            with open(os.path.join(tmp, fname), "wb") as f:
                f.write(fbytes)
        # Find the .shp file
        shp_files = [f for f in os.listdir(tmp) if f.lower().endswith(".shp")]
        if not shp_files:
            return None
        gdf = gpd.read_file(os.path.join(tmp, shp_files[0]))

    # Reproject to WGS84 if needed
    if gdf.crs is None:
        # Assume Lambert 93 if no CRS defined
        gdf = gdf.set_crs("EPSG:2154")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def shapefile_traces(gdf, color, name, opacity=0.5):
    """Convert a GeoDataFrame to Plotly Scattermap traces."""
    traces = []

    # Build a hover label from the first string column (if any)
    label_col = next((c for c in gdf.columns
                      if c != "geometry" and gdf[c].dtype == object), None)

    def hover(row):
        return str(row[label_col]) if label_col else ""

    def rgba(hex_color, a):
        r, g, b = int(hex_color[1:3],16), int(hex_color[3:5],16), int(hex_color[5:7],16)
        return f"rgba({r},{g},{b},{a})"

    color_solid = color                   # full opacity for outlines/lines
    color_fill  = rgba(color, opacity)    # user-controlled opacity for fills

    first = True
    for _, row in gdf.iterrows():
        geom = row.geometry
        lbl  = hover(row)
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type in ("Point", "MultiPoint"):
            pts = [geom] if geom.geom_type == "Point" else list(geom.geoms)
            traces.append(go.Scattermap(
                lat=[p.y for p in pts],
                lon=[p.x for p in pts],
                mode="markers",
                marker=dict(color=color_fill, size=8),
                name=name, hovertext=lbl, hoverinfo="text",
                legendgroup=name, showlegend=first,
            ))
            first = False

        elif geom.geom_type in ("LineString", "MultiLineString"):
            lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
            for line in lines:
                coords = list(line.coords)
                traces.append(go.Scattermap(
                    lat=[c[1] for c in coords] + [None],
                    lon=[c[0] for c in coords] + [None],
                    mode="lines",
                    line=dict(color=color_fill, width=2),
                    name=name, hovertext=lbl, hoverinfo="text",
                    legendgroup=name, showlegend=first,
                ))
                first = False

        elif geom.geom_type in ("Polygon", "MultiPolygon"):
            polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
            for poly in polys:
                coords = list(poly.exterior.coords)
                traces.append(go.Scattermap(
                    lat=[c[1] for c in coords] + [None],
                    lon=[c[0] for c in coords] + [None],
                    mode="lines",
                    fill="toself",
                    fillcolor=color_fill,
                    line=dict(color=color_solid, width=1.5),
                    name=name, hovertext=lbl, hoverinfo="text",
                    legendgroup=name, showlegend=first,
                ))
                first = False

    return traces
# ══════════════════════════════════════════════════════════════════════════════
def get_selected():
    return st.session_state.get("selected_point", None)

def highlight_marker_props(is_selected, base_color):
    """Return marker dict for a single point; highlighted ones get a bold ring."""
    if is_selected:
        return dict(color=base_color, size=14,
                    line=dict(color="#FFD700", width=3))
    return dict(color=base_color, size=9,
                line=dict(color="white", width=0.5))

# ══════════════════════════════════════════════════════════════════════════════
#  MAP
# ══════════════════════════════════════════════════════════════════════════════
def build_map(df, symbology, show_labels, selected_name, shp_layers=None):
    geo = df.dropna(subset=["_lon","_lat"]).copy()
    if geo.empty:
        return None

    groups  = get_groups(geo, symbology)
    fig     = go.Figure()

    # ── Shapefile layers (drawn first, underneath sample points) ──────────────
    if shp_layers:
        shp_colors  = st.session_state.get("shp_colors", {})
        shp_opacity = st.session_state.get("shp_opacity", {})
        for i, (layer_name, gdf) in enumerate(shp_layers):
            color   = shp_colors.get(layer_name, _SHP_COLORS[i % len(_SHP_COLORS)])
            opacity = shp_opacity.get(layer_name, 0.5)
            for t in shapefile_traces(gdf, color, layer_name, opacity):
                fig.add_trace(t)

    for g in groups:
        mask  = group_col(geo, symbology) == g
        sub   = geo[mask]
        c     = color_for(g, symbology, groups)
        lbl   = group_label(g, symbology)
        sym   = symbol_for(g, symbology, groups)

        is_sel = (sub["Name"] == selected_name) if selected_name else pd.Series(False, index=sub.index)

        # normal points
        normal = sub[~is_sel]
        if not normal.empty:
            fig.add_trace(go.Scattermap(
                lat=normal["_lat"], lon=normal["_lon"],
                mode="markers" + ("+text" if show_labels else ""),
                name=lbl,
                text=normal["Name"],
                textposition="top right",
                textfont=dict(size=8, color="#333"),
                customdata=normal["Name"],
                marker=dict(color=c, size=9, opacity=0.85),
                hovertemplate="<b>%{customdata}</b><br>lat: %{lat:.5f}<br>lon: %{lon:.5f}<extra>" + lbl + "</extra>",
                legendgroup=lbl,
            ))

        # highlighted point(s)
        sel_sub = sub[is_sel]
        if not sel_sub.empty:
            fig.add_trace(go.Scattermap(
                lat=sel_sub["_lat"], lon=sel_sub["_lon"],
                mode="markers+text",
                name=lbl,
                text=sel_sub["Name"],
                textposition="top right",
                textfont=dict(size=10, color="#111", family="Arial Black"),
                customdata=sel_sub["Name"],
                marker=dict(color="#FFD700", size=16,
                            allowoverlap=True),
                hovertemplate="<b>%{customdata}</b><br>lat: %{lat:.5f}<br>lon: %{lon:.5f}<extra>SELECTED</extra>",
                legendgroup=lbl,
                showlegend=False,
            ))

    center_lat = geo["_lat"].mean()
    center_lon = geo["_lon"].mean()

    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=9,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#ddd",
                    borderwidth=1, font=dict(size=11)),
        uirevision="map",   # preserve zoom/pan between re-renders
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  SCATTER HELPER
# ══════════════════════════════════════════════════════════════════════════════
def make_traces(d, x_vals, y_vals, symbology, show_labels, selected_name=None):
    groups = get_groups(d, symbology)
    traces = []
    for g in groups:
        mask  = group_col(d, symbology) == g
        xm    = x_vals[mask]; ym = y_vals[mask]
        nm    = d["Name"][mask]
        c     = color_for(g, symbology, groups)
        sym   = symbol_for(g, symbology, groups)
        lbl   = group_label(g, symbology)

        # Split into normal / selected
        sel_mask = (nm == selected_name) if selected_name else pd.Series(False, index=nm.index)

        normal_x  = xm[~sel_mask]; normal_y  = ym[~sel_mask]; normal_n  = nm[~sel_mask]
        sel_x     = xm[sel_mask];  sel_y     = ym[sel_mask];  sel_n     = nm[sel_mask]

        traces.append(go.Scatter(
            x=normal_x, y=normal_y,
            mode="markers+text" if show_labels else "markers",
            name=lbl,
            text=normal_n,
            customdata=normal_n,
            textposition="top right",
            textfont=dict(size=8, color="#333"),
            marker=dict(color=c, symbol=sym, size=9,
                        line=dict(color="white", width=0.5)),
            hovertemplate="<b>%{customdata}</b><br>x: %{x:.3g}<br>y: %{y:.3g}<extra>" + lbl + "</extra>",
            legendgroup=lbl,
        ))
        if not sel_x.empty:
            traces.append(go.Scatter(
                x=sel_x, y=sel_y,
                mode="markers+text",
                name=lbl,
                text=sel_n,
                customdata=sel_n,
                textposition="top right",
                textfont=dict(size=10, color="#111", family="Arial Black"),
                marker=dict(color="#FFD700", symbol=sym, size=15,
                            line=dict(color="#B8860B", width=2)),
                hovertemplate="<b>%{customdata}</b><br>x: %{x:.3g}<br>y: %{y:.3g}<extra>SELECTED</extra>",
                legendgroup=lbl,
                showlegend=False,
            ))
    return traces

def base_fig(title, xlab, yl):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#111"), x=0.04),
        xaxis_title=xlab, yaxis_title=yl,
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#ddd",
                    borderwidth=1, font=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="DejaVu Sans, Arial, sans-serif", size=12),
        margin=dict(l=60, r=30, t=50, b=60),
        hovermode="closest",
        uirevision=title,   # preserve zoom between re-renders
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False,
                     linecolor="#ccc", ticks="outside")
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False,
                     linecolor="#ccc", ticks="outside")
    return fig

def add_line(fig, x0, x1, y0, y1, color="#aaa", dash="dash", name="", width=1.2):
    fig.add_trace(go.Scatter(
        x=[x0,x1], y=[y0,y1], mode="lines",
        line=dict(color=color, dash=dash, width=width),
        name=name, showlegend=bool(name), hoverinfo="skip",
    ))

def add_model_areas(fig):
    slope    = (50-100)/(-11-(-25))
    y_at_x15 = 100 + slope*(-15-(-25))
    fig.add_trace(go.Scatter(
        x=[-25,-15,-15,-25,-25], y=[0,0,y_at_x15,100,0],
        fill="toself", fillcolor="rgba(74,144,217,0.15)",
        line=dict(color="#4A90D9",width=1.2),
        name="Pearson model", hoverinfo="skip", showlegend=False))
    fig.add_annotation(x=-20,y=25,text="<i>Pearson<br>model</i>",
                       showarrow=False,font=dict(color="#1a5fa8",size=9))
    fig.add_trace(go.Scatter(
        x=[-11,0,-11,-11],y=[0,0,50,0],
        fill="toself",fillcolor="rgba(232,168,56,0.15)",
        line=dict(color="#E8A838",width=1.2),
        name="Eichinger model",hoverinfo="skip",showlegend=False))
    fig.add_annotation(x=-5.5,y=12,text="<i>Eichinger<br>model</i>",
                       showarrow=False,font=dict(color="#a06010",size=9))
    fig.add_trace(go.Scatter(
        x=[-15,-11,-11,-15,-15],y=[0,0,50,100,0],
        fill="toself",fillcolor="rgba(232,112,64,0.15)",
        line=dict(color="#E87040",width=1.2),
        name="Mook model",hoverinfo="skip",showlegend=False))
    fig.add_annotation(x=-13,y=25,text="<i>Mook<br>model</i>",
                       showarrow=False,font=dict(color="#b84a10",size=9))

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT FUNCTIONS — data logic taken verbatim from hydrochem_plots.py
#  Only rendering is adapted from matplotlib → Plotly.
# ══════════════════════════════════════════════════════════════════════════════

# ── PLOT 1 — CAI ─────────────────────────────────────────────────────────────
def plot_cai(df, symbology, show_labels, sel):
    req = ["Na_meq","K_meq","Cl_meq","SO4_meq","HCO3_meq"]
    d = df.dropna(subset=req).copy()
    if d.empty: return None
    num  = d["Cl_meq"] - (d["Na_meq"] + d["K_meq"])
    x = num / d["Cl_meq"].replace(0, np.nan)
    y = num / (d["SO4_meq"] + d["HCO3_meq"]).replace(0, np.nan)
    fig = base_fig("Chloro-Alkaline Indices", "CAI-1", "CAI-2")
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    fig.add_hline(y=0, line_color="#999", line_dash="dash", line_width=1.0)
    fig.add_vline(x=0, line_color="#999", line_dash="dash", line_width=1.0)
    return fig

# ── PLOT 2 — (Ca²⁺ + Mg²⁺) vs HCO₃⁻  (Na-normalised) ──────────────────────
def plot_na_norm(df, symbology, show_labels, sel):
    req = ["Na_meq","Ca_meq","Mg_meq","HCO3_meq"]
    d = df.dropna(subset=req).copy()
    if d.empty: return None
    x = (d["Ca_meq"] + d["Mg_meq"]) / d["Na_meq"].replace(0, np.nan)
    y = d["HCO3_meq"] / d["Na_meq"].replace(0, np.nan)
    fig = base_fig("(Ca²⁺ + Mg²⁺) vs HCO₃⁻  (Na-normalised, meq/L)",
                   "(Ca²⁺+Mg²⁺)/Na⁺  (meq/L)", "HCO₃⁻/Na⁺  (meq/L)")
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    hi = max(x.max(), y.max()) * 1.05
    add_line(fig, 0, hi, 0, hi, "#aaa", "dash", "1:1")
    fig.update_xaxes(range=[0, None]); fig.update_yaxes(range=[0, None])
    return fig

# ── PLOT 3 — (Ca²⁺ + Mg²⁺) vs HCO₃⁻  (unnormalised) ───────────────────────
def plot_hco3_camg(df, symbology, show_labels, sel):
    req = ["Ca_meq","Mg_meq","HCO3_meq"]
    d = df.dropna(subset=req).copy()
    if d.empty: return None
    x = d["Ca_meq"] + d["Mg_meq"]
    y = d["HCO3_meq"]
    fig = base_fig("(Ca²⁺ + Mg²⁺) vs HCO₃⁻  (unnormalised, meq/L)",
                   "Ca²⁺ + Mg²⁺  (meq/L)", "HCO₃⁻  (meq/L)")
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    hi = max(x.max(), y.max()) * 1.08
    add_line(fig, 0, hi, 0, hi, "#aaa", "dash", "1:1")
    fig.update_xaxes(range=[0, None]); fig.update_yaxes(range=[0, None])
    return fig

# ── PLOT 4 — Cl vs Na/Cl ─────────────────────────────────────────────────────
def plot_nacl(df, symbology, show_labels, sel):
    req = ["Na_meq","Cl_meq"]
    d = df.dropna(subset=req).copy()
    if d.empty: return None
    x = d["Cl_meq"]
    y = d["Na_meq"] / d["Cl_meq"].replace(0, np.nan)
    fig = base_fig("Cl vs Na/Cl", "Cl⁻  (meq/L)", "Na⁺/Cl⁻  (meq/L)")
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    fig.add_hline(y=1, line_color="#555", line_dash="dash", line_width=1.0,
                  annotation_text="Na/Cl = 1", annotation_font_size=10)
    return fig

# ── PLOT 5 — Cl vs Ca/Cl ─────────────────────────────────────────────────────
def plot_cacl(df, symbology, show_labels, sel):
    req = ["Ca_meq","Cl_meq"]
    d = df.dropna(subset=req).copy()
    if d.empty: return None
    x = d["Cl_meq"]
    y = d["Ca_meq"] / d["Cl_meq"].replace(0, np.nan)
    fig = base_fig("Cl vs Ca/Cl", "Cl⁻  (meq/L)", "Ca²⁺/Cl⁻  (meq/L)")
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    return fig

# ── PLOT 6 — TDS vs EC ───────────────────────────────────────────────────────
def plot_ec_tds(df, COL, symbology, show_labels, sel):
    ce_col  = COL.get("CE")
    tds_col = COL.get("TDS")
    if not ce_col or not tds_col or ce_col not in df.columns or tds_col not in df.columns: return None
    d = df.dropna(subset=[ce_col, tds_col]).copy()
    d["CE_num"]  = pd.to_numeric(d[ce_col],  errors="coerce")
    d["TDS_num"] = pd.to_numeric(d[tds_col], errors="coerce")
    d = d.dropna(subset=["CE_num","TDS_num"])
    if d.empty: return None
    fig = base_fig("TDS vs EC", "TDS  (mg/L)", "EC  (µS/cm)")
    for t in make_traces(d, d["TDS_num"], d["CE_num"], symbology, show_labels, sel): fig.add_trace(t)
    return fig

# ── PLOT 7 — (Na+K)−Cl  vs  (Ca+Mg)−(HCO3+SO4)  cation exchange ─────────────
def plot_exchange_balance(df, symbology, show_labels, sel):
    req = ["Na_meq","K_meq","Cl_meq","Ca_meq","Mg_meq","HCO3_meq","SO4_meq"]
    d = df.dropna(subset=req).copy()
    if d.empty: return None
    x = (d["Na_meq"] + d["K_meq"]) - d["Cl_meq"]
    y = (d["Ca_meq"] + d["Mg_meq"]) - (d["HCO3_meq"] + d["SO4_meq"])
    fig = base_fig("Cation exchange balance",
                   "(Na⁺+K⁺) − Cl⁻  (meq/L)",
                   "(Ca²⁺+Mg²⁺) − (HCO₃⁻+SO₄²⁻)  (meq/L)")
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    lim = max(abs(x).max(), abs(y).max()) * 1.15
    add_line(fig, -lim, lim, lim, -lim, "black", "dash", "slope −1")
    fig.add_hline(y=0, line_color="#ddd", line_width=0.7)
    fig.add_vline(x=0, line_color="#ddd", line_width=0.7)
    return fig

# ── PLOT 8 — pH vs HCO₃⁻ (mmol/L) ──────────────────────────────────────────
def plot_hco3_vs_ph(df, COL, symbology, show_labels, sel):
    ph_col  = COL.get("pH")
    hco_col = COL.get("HCO3")
    if not ph_col or not hco_col or ph_col not in df.columns or hco_col not in df.columns: return None
    d = df.dropna(subset=[ph_col, hco_col]).copy()
    if d.empty: return None
    x = pd.to_numeric(d[ph_col],  errors="coerce")
    y = pd.to_numeric(d[hco_col], errors="coerce") / MW["HCO3"]
    fig = base_fig("pH vs HCO₃⁻", "pH", "HCO₃  (mmol/L)")
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    return fig

# ── PLOT 9 — Cl⁻ vs Na⁺ (meq/L) ─────────────────────────────────────────────
def plot_na_vs_cl(df, symbology, show_labels, sel):
    req = ["Na_meq","Cl_meq"]
    d = df.dropna(subset=req).copy()
    if d.empty: return None
    x = d["Cl_meq"]; y = d["Na_meq"]
    fig = base_fig("Cl⁻ vs Na⁺", "Cl⁻  (meq/L)", "Na⁺  (meq/L)")
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    hi = max(x.max(), y.max()) * 1.08
    add_line(fig, 0, hi, 0, hi, "#aaa", "dash", "1:1")
    fig.update_xaxes(range=[0, None]); fig.update_yaxes(range=[0, None])
    return fig

# ── PLOT 10 — Cl⁻ vs Ca²⁺ (meq/L) ──────────────────────────────────────────
def plot_ca_vs_cl(df, symbology, show_labels, sel):
    req = ["Ca_meq","Cl_meq"]
    d = df.dropna(subset=req).copy()
    if d.empty: return None
    x = d["Cl_meq"]; y = d["Ca_meq"]
    fig = base_fig("Cl⁻ vs Ca²⁺", "Cl⁻  (meq/L)", "Ca²⁺  (meq/L)")
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    return fig

# ── PLOT 11 — Cl⁻ vs SO₄²⁻ (meq/L) ─────────────────────────────────────────
def plot_so4_vs_cl(df, symbology, show_labels, sel):
    req = ["SO4_meq","Cl_meq"]
    d = df.dropna(subset=req).copy()
    if d.empty: return None
    x = d["Cl_meq"]; y = d["SO4_meq"]
    fig = base_fig("Cl⁻ vs SO₄²⁻", "Cl⁻  (meq/L)", "SO₄²⁻  (meq/L)")
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    return fig

# ── PLOT 12 — SO₄²⁻ vs Ca²⁺ (meq/L) ─────────────────────────────────────────
def plot_ca_vs_so4(df, symbology, show_labels, sel):
    req = ["Ca_meq","SO4_meq"]
    d = df.dropna(subset=req).copy()
    if d.empty: return None
    x = d["SO4_meq"]; y = d["Ca_meq"]
    fig = base_fig("SO₄²⁻ vs Ca²⁺", "SO₄²⁻  (meq/L)", "Ca²⁺  (meq/L)")
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    hi = max(x.max(), y.max()) * 1.08
    add_line(fig, 0, hi, 0, hi, "#aaa", "dash", "1:1")
    fig.update_xaxes(range=[0, None]); fig.update_yaxes(range=[0, None])
    return fig

# ── PLOT 13 — log pCO₂ vs HCO₃⁻ and Ca²⁺ ───────────────────────────────────
def plot_pco2(df, COL, symbology, show_labels, sel):
    pc = COL.get("PCO2")
    if not pc or pc not in df.columns: return None
    req = ["Ca_mmol","HCO3_mmol"]
    d = df.dropna(subset=req + [pc]).copy()
    if d.empty: return None
    logp = pd.to_numeric(d[pc], errors="coerce")
    # PHREEQC open-system calcite dissolution at 15°C
    pco2_ref = [-1.5, -2.0, -2.5, -3.0, -3.5]
    ca_ref   = [2.96,  1.95,  1.29,  0.86,  0.58]
    alk_ref  = [5.79,  3.82,  2.53,  1.68,  1.10]
    fig = base_fig("log pCO₂ vs HCO₃⁻ and Ca²⁺",
                   "log pCO₂  (atm)", "Concentration  (mmol/L)")
    fig.add_trace(go.Scatter(x=pco2_ref, y=ca_ref, mode="lines+markers",
        name="Ca²⁺ open system 15°C (PHREEQC)",
        line=dict(color="black", width=1.6),
        marker=dict(symbol="circle", size=5, color="black")))
    fig.add_trace(go.Scatter(x=pco2_ref, y=alk_ref, mode="lines+markers",
        name="HCO₃⁻ open system 15°C (PHREEQC)",
        line=dict(color="black", width=1.6, dash="dash"),
        marker=dict(symbol="square", size=5, color="black")))
    groups = get_groups(d, symbology)
    for g in groups:
        mask = group_col(d, symbology) == g
        c    = color_for(g, symbology, groups)
        sym  = symbol_for(g, symbology, groups)
        lbl  = group_label(g, symbology)
        is_sel = (d["Name"][mask] == sel) if sel else pd.Series(False, index=d[mask].index)
        fig.add_trace(go.Scatter(
            x=logp[mask], y=d["Ca_mmol"][mask], mode="markers", name=lbl,
            text=d["Name"][mask], customdata=d["Name"][mask], legendgroup=lbl,
            marker=dict(color=np.where(is_sel,"#FFD700",c), symbol=sym,
                        size=np.where(is_sel,14,9),
                        line=dict(color=np.where(is_sel,"#B8860B","white"),
                                  width=np.where(is_sel,2,0.5))),
            hovertemplate="<b>%{customdata}</b><br>pCO₂: %{x:.3g}<br>Ca: %{y:.3g}<extra>"+lbl+" Ca²⁺</extra>"))
        fig.add_trace(go.Scatter(
            x=logp[mask], y=d["HCO3_mmol"][mask], mode="markers", name=lbl,
            text=d["Name"][mask], customdata=d["Name"][mask],
            legendgroup=lbl, showlegend=False,
            marker=dict(color="rgba(0,0,0,0)", symbol=sym, size=9,
                        line=dict(color=c, width=1.6)),
            hovertemplate="<b>%{customdata}</b><br>pCO₂: %{x:.3g}<br>HCO₃: %{y:.3g}<extra>"+lbl+" HCO₃⁻</extra>"))
    return fig

# ── PLOT 14 — log pCO₂ vs SI Calcite ────────────────────────────────────────
def plot_si_pco2(df, COL, symbology, show_labels, sel):
    pc  = COL.get("PCO2")
    sic = COL.get("siCal")
    if not pc or not sic or pc not in df.columns or sic not in df.columns: return None
    d = df.dropna(subset=[pc, sic]).copy()
    if d.empty: return None
    x = pd.to_numeric(d[pc],  errors="coerce")
    y = pd.to_numeric(d[sic], errors="coerce")
    fig = base_fig("log pCO₂ vs SI Calcite", "log pCO₂  (atm)", "SI Calcite")
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    fig.add_hline(y=0, line_color="#555", line_dash="dash", line_width=1.0,
                  annotation_text="SI = 0  (equilibrium)", annotation_font_size=10)
    return fig

# ── PLOT 15 — log pCO₂ vs pH ────────────────────────────────────────────────
def plot_pco2_vs_ph(df, COL, symbology, show_labels, sel):
    pc_col = COL.get("PCO2")
    ph_col = COL.get("pH")
    if not pc_col or not ph_col or pc_col not in df.columns or ph_col not in df.columns: return None
    d = df.dropna(subset=[pc_col, ph_col]).copy()
    if d.empty: return None
    x = pd.to_numeric(d[pc_col], errors="coerce")
    y = pd.to_numeric(d[ph_col], errors="coerce")
    fig = base_fig("log pCO₂ vs pH", "log pCO₂  (atm)", "pH")
    fig.add_vline(x=-2.5, line_color="#aaa", line_dash="dot", line_width=0.8,
                  annotation_text=" soil CO₂",      annotation_font_size=8)
    fig.add_vline(x=-3.5, line_color="#aaa", line_dash="dot", line_width=0.8,
                  annotation_text=" atmosphere",     annotation_font_size=8)
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    return fig

# ── PLOT 16 — δ¹³C vs ¹⁴C — Diagnostic areas ───────────────────────────────
def plot_14c_13c_areas(df, COL, symbology, show_labels, sel):
    c14_col = COL.get("C14")
    c13_col = COL.get("C13")
    if not c14_col or not c13_col or c14_col not in df.columns or c13_col not in df.columns: return None
    d = df.dropna(subset=[c14_col, c13_col]).copy()
    if d.empty: return None
    x = pd.to_numeric(d[c13_col], errors="coerce")
    y = pd.to_numeric(d[c14_col], errors="coerce")
    fig = base_fig("δ¹³C vs ¹⁴C — Diagnostic areas", "δ¹³C  (‰ VPDB)", "¹⁴C  (pmc)")
    add_model_areas(fig)
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    fig.update_xaxes(range=[-30, 5]); fig.update_yaxes(range=[0, 110])
    return fig

# ── PLOT 17 — δ¹³C vs ¹⁴C — Mixing lines ───────────────────────────────────
def plot_and_d13c_lines(df, COL, symbology, show_labels, sel):
    c14_col = COL.get("C14")
    c13_col = COL.get("C13")
    if not c14_col or not c13_col or c14_col not in df.columns or c13_col not in df.columns: return None
    d = df.dropna(subset=[c14_col, c13_col]).copy()
    if d.empty: return None
    x = pd.to_numeric(d[c13_col], errors="coerce")
    y = pd.to_numeric(d[c14_col], errors="coerce")
    fig = base_fig("δ¹³C vs ¹⁴C — Mixing lines", "δ¹³C  (‰ VPDB)", "¹⁴C  (pmc)")
    fig.add_trace(go.Scatter(x=[-25,0], y=[100,0], mode="lines",
        name="soil CO₂ C3 cover (δ¹³C = −25‰)",
        line=dict(color="#185FA5", width=1.5)))
    fig.add_trace(go.Scatter(x=[-15,0], y=[100,0], mode="lines",
        name="soil CO₂ C4 cover (δ¹³C = −15‰)",
        line=dict(color="#D85A30", width=1.5)))
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    fig.update_xaxes(range=[-30, 5]); fig.update_yaxes(range=[0, 110])
    return fig

# ── PLOT 18 — δ¹³C vs ¹⁴C — C3 Pearson Mixing line ────────────────────────
def plot_14c_13c_pearson(df, COL, symbology, show_labels, sel):
    c14_col = COL.get("C14")
    c13_col = COL.get("C13")
    if not c14_col or not c13_col or c14_col not in df.columns or c13_col not in df.columns: return None
    d = df.dropna(subset=[c14_col, c13_col]).copy()
    if d.empty: return None
    x = pd.to_numeric(d[c13_col], errors="coerce")
    y = pd.to_numeric(d[c14_col], errors="coerce")
    fig = base_fig("¹⁴C vs δ¹³C — Pearson mixing line", "δ¹³C  (‰ VPDB)", "¹⁴C  (pmc)")
    fig.add_trace(go.Scatter(x=[-25, 0], y=[100, 0], mode="lines+markers",
        name="Pearson mixing line",
        line=dict(color="black", width=1.5),
        marker=dict(color="black", size=9)))
    fig.add_annotation(x=-26.5, y=101.5, text="Soil CO₂",
                       showarrow=False, font=dict(color="black", size=9, weight=700))
    fig.add_annotation(x=0.3,   y=1.5,   text="Carbonate",
                       showarrow=False, font=dict(color="black", size=9, weight=700))
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    fig.update_xaxes(range=[-30, 5]); fig.update_yaxes(range=[0, 110])
    return fig

# ── PLOT 19 — δ¹³C observed vs δ¹³C computed NETPATH ───────────────────────
def plot_d13c_comp_vs_obs(df, COL, symbology, show_labels, sel):
    c13c_col = COL.get("C13c")
    c13_col  = COL.get("C13")
    if not c13c_col or not c13_col or c13c_col not in df.columns or c13_col not in df.columns: return None
    d = df.dropna(subset=[c13c_col, c13_col]).copy()
    if d.empty: return None
    x = pd.to_numeric(d[c13_col],  errors="coerce")   # observed → x
    y = pd.to_numeric(d[c13c_col], errors="coerce")   # computed → y
    fig = base_fig("δ¹³C observed vs δ¹³C computed NETPATH",
                   "δ¹³C observed  (‰ VPDB)", "δ¹³C computed  (‰ VPDB)")
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    lo = min(x.min(), y.min()) - 1; hi = max(x.max(), y.max()) + 1
    add_line(fig, lo, hi, lo, hi, "black", "dash", "1:1")
    return fig

# ── PLOT 20 — ¹⁴C Aobs vs ¹⁴C Acorr NETPATH ────────────────────────────────
def plot_acorr_vs_14c(df, COL, symbology, show_labels, sel):
    acorr_col = COL.get("Acorr")
    c14_col   = COL.get("C14")
    if not acorr_col or not c14_col or acorr_col not in df.columns or c14_col not in df.columns: return None
    d = df.dropna(subset=[acorr_col, c14_col]).copy()
    if d.empty: return None
    x = pd.to_numeric(d[c14_col],   errors="coerce")   # measured ¹⁴C  → x
    y = pd.to_numeric(d[acorr_col], errors="coerce")   # corrected NETPATH → y
    fig = base_fig("¹⁴C Aobs vs Acorr NETPATH",
                   "¹⁴C observed  (pmc)", "¹⁴C corrected  (pmc)  — NETPATH")
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    hi = max(x.max(), y.max()) * 1.08
    add_line(fig, 0, hi, 0, hi, "black", "dash", "1:1")
    fig.update_xaxes(range=[0, None]); fig.update_yaxes(range=[0, None])
    return fig

# ── PLOT 21 — δ¹³C vs ¹⁴C And NETPATH ──────────────────────────────────────
def plot_and_d13c(df, COL, symbology, show_labels, sel):
    and_col = COL.get("And")
    c13_col = COL.get("C13")
    if not and_col or not c13_col or and_col not in df.columns or c13_col not in df.columns: return None
    d = df.dropna(subset=[and_col, c13_col]).copy()
    if d.empty: return None
    x = pd.to_numeric(d[c13_col], errors="coerce")
    y = pd.to_numeric(d[and_col], errors="coerce")
    fig = base_fig("δ¹³C vs ¹⁴C And NETPATH",
                   "δ¹³C  (‰ VPDB)", "¹⁴C  (pmc)  — no decay")
    fig.add_trace(go.Scatter(x=[-25,0], y=[100,0], mode="lines",
        name="soil CO₂ C3 cover (δ¹³C = −25‰)",
        line=dict(color="#185FA5", width=1.5)))
    fig.add_trace(go.Scatter(x=[-15,0], y=[100,0], mode="lines",
        name="soil CO₂ C4 cover (δ¹³C = −15‰)",
        line=dict(color="#D85A30", width=1.5)))
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    fig.update_xaxes(range=[-30, 5]); fig.update_yaxes(range=[0, 110])
    return fig

# ── PLOT 22 — δ¹³C vs ¹⁴C Acorr NETPATH ────────────────────────────────────
def plot_acorr_d13c_lines(df, COL, symbology, show_labels, sel):
    acorr_col = COL.get("Acorr")
    c13_col   = COL.get("C13")
    if not acorr_col or not c13_col or acorr_col not in df.columns or c13_col not in df.columns: return None
    d = df.dropna(subset=[acorr_col, c13_col]).copy()
    if d.empty: return None
    x = pd.to_numeric(d[c13_col],   errors="coerce")
    y = pd.to_numeric(d[acorr_col], errors="coerce")
    fig = base_fig("δ¹³C vs ¹⁴C Acorr NETPATH",
                   "δ¹³C  (‰ VPDB)", "¹⁴C  (pmc)  — Corrected NETPATH")
    fig.add_trace(go.Scatter(x=[-25,0], y=[100,0], mode="lines",
        name="soil CO₂ C3 cover (δ¹³C = −25‰)",
        line=dict(color="#185FA5", width=1.5)))
    fig.add_trace(go.Scatter(x=[-15,0], y=[100,0], mode="lines",
        name="soil CO₂ C4 cover (δ¹³C = −15‰)",
        line=dict(color="#D85A30", width=1.5)))
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    fig.update_xaxes(range=[-30, 5]); fig.update_yaxes(range=[0, 110])
    return fig

# ── PLOT 23 — ¹⁴C Acorr - NETPATH vs Pearson ───────────────────────────────
def plot_acorr_comparison(df, COL, symbology, show_labels, sel):
    np_col = COL.get("Acorr")
    pr_col = COL.get("AcorrP")
    if not np_col or not pr_col or np_col not in df.columns or pr_col not in df.columns: return None
    d = df.dropna(subset=[np_col, pr_col]).copy()
    if d.empty: return None
    x = pd.to_numeric(d[np_col], errors="coerce")
    y = pd.to_numeric(d[pr_col], errors="coerce")
    fig = base_fig("Corrected ¹⁴C activity - NETPATH vs Pearson",
                   "¹⁴C corrected — NETPATH  (pmc)",
                   "¹⁴C corrected — Pearson  (pmc)")
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    lo = min(x.min(), y.min()) * 0.95; hi = max(x.max(), y.max()) * 1.05
    add_line(fig, lo, hi, lo, hi, "black", "dash", "1:1")
    return fig

# ── PLOT 24 — Radiocarbon age - NETPATH vs Pearson ──────────────────────────
def plot_age_comparison(df, COL, symbology, show_labels, sel):
    np_col = COL.get("AgeNP")
    pr_col = COL.get("AgePR")
    if not np_col or not pr_col or np_col not in df.columns or pr_col not in df.columns: return None
    d = df.dropna(subset=[np_col, pr_col]).copy()
    if d.empty: return None
    x = pd.to_numeric(d[np_col], errors="coerce")
    y = pd.to_numeric(d[pr_col], errors="coerce")
    fig = base_fig("Radiocarbon age - NETPATH vs Pearson",
                   "Model age — NETPATH  (years BP)",
                   "Model age — Pearson  (years BP)")
    for t in make_traces(d, x, y, symbology, show_labels, sel): fig.add_trace(t)
    lo = min(0, min(x.min(), y.min()) * 0.95); hi = max(x.max(), y.max()) * 1.05
    add_line(fig, lo, hi, lo, hi, "black", "dash", "1:1")
    return fig

# ── PLOT 25 — Acorr - Pearson / NETPATH ratio (bar) ─────────────────────────
def plot_ratio_bar(df, COL, symbology, sel):
    np_col = COL.get("Acorr")
    pr_col = COL.get("AcorrP")
    bh_col = COL.get("Borehole")
    if not np_col or not pr_col or np_col not in df.columns or pr_col not in df.columns: return None
    d = df.dropna(subset=[np_col, pr_col]).copy()
    if d.empty: return None
    d["_label"] = d[bh_col].astype(str) if bh_col and bh_col in df.columns else d["Name"].astype(str)
    d["_ratio"] = pd.to_numeric(d[pr_col], errors="coerce") / pd.to_numeric(d[np_col], errors="coerce")
    d = d.dropna(subset=["_ratio"]).sort_values("_ratio")
    groups  = get_groups(d, symbology)
    colors  = [color_for(group_col(d, symbology).iloc[i], symbology, groups) for i in range(len(d))]
    colors  = [c if d["Name"].iloc[i] != sel else "#FFD700" for i, c in enumerate(colors)]
    fig = go.Figure(go.Bar(x=d["_label"], y=d["_ratio"], marker_color=colors,
        marker_line_color="white", marker_line_width=0.5,
        customdata=d["Name"],
        hovertemplate="<b>%{customdata}</b><br>Ratio: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=1.0, line_color="black", line_dash="dash", line_width=1.2,
                  annotation_text="Ratio = 1  (models agree)")
    fig.update_layout(
        title="Acorr - Pearson / NETPATH ratio", xaxis_title="",
        yaxis_title="Pearson ¹⁴C corrected / NETPATH ¹⁴C corrected",
        xaxis_tickangle=-45, plot_bgcolor="white", paper_bgcolor="white",
        uirevision="ratio_bar")
    return fig

# ── PLOT 26 — Radiocarbon age - Pearson / NETPATH ratio (bar) ────────────────
def plot_age_ratio_bar(df, COL, symbology, sel):
    np_col = COL.get("AgeNP")
    pr_col = COL.get("AgePR")
    bh_col = COL.get("Borehole")
    if not np_col or not pr_col or np_col not in df.columns or pr_col not in df.columns: return None
    d = df.dropna(subset=[np_col, pr_col]).copy()
    d["_np"] = pd.to_numeric(d[np_col], errors="coerce")
    d["_pr"] = pd.to_numeric(d[pr_col], errors="coerce")
    d = d.dropna(subset=["_np","_pr"])
    d = d[d["_np"] != 0]   # avoid division by zero
    if d.empty: return None
    d["_ratio"] = d["_pr"] / d["_np"]
    d = d.sort_values("_ratio")
    d["_label"] = d[bh_col].astype(str) if bh_col and bh_col in df.columns else d["Name"].astype(str)
    groups  = get_groups(d, symbology)
    colors  = [color_for(group_col(d, symbology).iloc[i], symbology, groups) for i in range(len(d))]
    colors  = [c if d["Name"].iloc[i] != sel else "#FFD700" for i, c in enumerate(colors)]
    fig = go.Figure(go.Bar(x=d["_label"], y=d["_ratio"], marker_color=colors,
        marker_line_color="white", marker_line_width=0.5,
        customdata=d["Name"],
        hovertemplate="<b>%{customdata}</b><br>Age ratio: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=1.0, line_color="black", line_dash="dash", line_width=1.2,
                  annotation_text="Ratio = 1  (models agree)")
    fig.update_layout(
        title="Radiocarbon age - Pearson / NETPATH ratio", xaxis_title="",
        yaxis_title="Pearson age / NETPATH age",
        xaxis_tickangle=-45, plot_bgcolor="white", paper_bgcolor="white",
        uirevision="age_ratio_bar")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
def _w(fn, needs_COL=True, is_bar=False):
    def builder(df, COL, sym, labels, sel):
        if is_bar:  return fn(df, COL, sym, sel)
        if needs_COL: return fn(df, COL, sym, labels, sel)
        return fn(df, sym, labels, sel)
    return builder

PLOTS = [
    ("1. Chloro-Alkaline Indices",                                          _w(plot_cai, False)),
    ("2. (Ca²⁺ + Mg²⁺) vs HCO₃⁻  (Na-normalised, meq/L)",                _w(plot_na_norm, False)),
    ("3. (Ca²⁺ + Mg²⁺) vs HCO₃⁻  (unnormalised, meq/L)",                 _w(plot_hco3_camg, False)),
    ("4. Cl vs Na/Cl",                                                      _w(plot_nacl, False)),
    ("5. Cl vs Ca/Cl",                                                      _w(plot_cacl, False)),
    ("6. TDS vs EC",                                                        _w(plot_ec_tds)),
    ("7. (Na+K)−Cl  vs  (Ca+Mg)−(HCO3+SO4)  — cation exchange diagnostic",_w(plot_exchange_balance, False)),
    ("8. pH vs HCO₃⁻  (mmol/L)",                                           _w(plot_hco3_vs_ph)),
    ("9. Cl⁻ vs Na⁺ (meq/L)",                                              _w(plot_na_vs_cl, False)),
    ("10. Cl⁻ vs Ca²⁺ (meq/L)",                                            _w(plot_ca_vs_cl, False)),
    ("11. Cl⁻ vs SO₄²⁻ (meq/L)",                                          _w(plot_so4_vs_cl, False)),
    ("12. SO₄²⁻ vs Ca²⁺ (meq/L)",                                         _w(plot_ca_vs_so4, False)),
    ("13. log pCO₂ vs HCO₃⁻ and Ca²⁺",                                    _w(plot_pco2)),
    ("14. log pCO₂ vs SI Calcite",                                          _w(plot_si_pco2)),
    ("15. log pCO₂ vs pH",                                                  _w(plot_pco2_vs_ph)),
    ("16. δ¹³C vs ¹⁴C — Diagnostic areas",                                 _w(plot_14c_13c_areas)),
    ("17. δ¹³C vs ¹⁴C — Mixing lines",                                     _w(plot_and_d13c_lines)),
    ("18. ¹⁴C vs δ¹³C — Pearson mixing line",                              _w(plot_14c_13c_pearson)),
    ("19. δ¹³C observed vs δ¹³C computed NETPATH",                         _w(plot_d13c_comp_vs_obs)),
    ("20. ¹⁴C Aobs vs Acorr NETPATH",                                      _w(plot_acorr_vs_14c)),
    ("21. δ¹³C vs ¹⁴C And NETPATH",                                        _w(plot_and_d13c)),
    ("22. δ¹³C vs ¹⁴C Acorr NETPATH",                                      _w(plot_acorr_d13c_lines)),
    ("23. Corrected ¹⁴C activity - NETPATH vs Pearson",                    _w(plot_acorr_comparison)),
    ("24. Radiocarbon age - NETPATH vs Pearson",                            _w(plot_age_comparison)),
    ("25. Acorr - Pearson / NETPATH ratio",                                 _w(plot_ratio_bar, is_bar=True)),
    ("26. Radiocarbon age - Pearson / NETPATH ratio",                       _w(plot_age_ratio_bar, is_bar=True)),
]

# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Hydrochem Dashboard", page_icon="💧", layout="wide")

# ── CSS: sticky top / scrollable bottom split ─────────────────────────────────
st.markdown("""
<style>
/* ── global resets ── */
[data-testid="stSidebar"] { background: #f0f4f8; }
h1 { color: #185FA5; }

/* Hide default streamlit top padding so our sticky panel can sit flush */
.block-container { padding-top: 0 !important; padding-bottom: 0 !important; }

/* ── sticky map wrapper ── */
#map-sticky-wrapper {
    position: sticky;
    top: 0;
    z-index: 100;
    background: white;
    border-bottom: 2px solid #e0e8f0;
    padding: 8px 0 4px 0;
}

/* ── selected-point banner ── */
.selected-banner {
    background: #FFF8DC; border: 1.5px solid #DAA520;
    border-radius: 8px; padding: 5px 14px;
    font-weight: 600; color: #7a5c00;
    display: inline-block; margin: 4px 0 6px 0;
}

/* ── plot grid: two per row, square-ish ── */
.plot-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    padding: 12px 0 40px 0;
}
.plot-cell {
    background: white;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    overflow: hidden;
}

/* metrics row */
.metric-row {
    display: flex; gap: 16px; margin: 6px 0 10px 0;
}
.metric-box {
    background: #f0f4f8; border-radius: 8px;
    padding: 8px 18px; text-align: center; flex: 1;
}
.metric-box .val { font-size: 1.5em; font-weight: 700; color: #185FA5; }
.metric-box .lbl { font-size: 0.78em; color: #555; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "selected_point" not in st.session_state:
    st.session_state["selected_point"] = None
if "shp_uploader_key" not in st.session_state:
    st.session_state["shp_uploader_key"] = 0
if "shp_layers" not in st.session_state:
    st.session_state["shp_layers"] = []
if "shp_colors" not in st.session_state:
    st.session_state["shp_colors"] = {}
if "shp_opacity" not in st.session_state:
    st.session_state["shp_opacity"] = {}
if "custom_plot_active" not in st.session_state:
    st.session_state["custom_plot_active"] = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload CSV", type=["csv","CSV"])

    st.divider()
    st.subheader("Display")
    symbology   = st.radio("Colour scheme", ["AQUIFER","CLUSTER"], horizontal=True)
    show_labels = st.toggle("Show sample labels", value=True)
    label_column = None
    if show_labels and uploaded:
        import chardet as _cd
        _raw  = uploaded.getvalue()
        _enc  = _cd.detect(_raw[:20000])["encoding"] or "latin-1"
        _text = _raw.decode(_enc, errors="replace")
        _sep  = ";" if _text.split("\n")[0].count(";") > _text.split("\n")[0].count(",") else ","
        _cols = pd.read_csv(io.StringIO(_text), sep=_sep, nrows=0).columns.str.strip().tolist()
        _name_candidates = ["name","sample","sample name","sample id","sampleid","id","code","borehole","well"]
        _low  = {c.lower().strip(): c for c in _cols}
        _default = next((c for n in _name_candidates for c in [_low.get(n)] if c), _cols[0])
        label_column = st.selectbox("Label column", options=_cols, index=_cols.index(_default) if _default in _cols else 0)

    st.divider()
    st.subheader("🔍 Filter rows")
    filter_on = st.toggle("Enable filter", value=False)
    filter_col_name = None; filter_vals = []
    if filter_on and uploaded:
        import chardet as _cd2
        _raw2  = uploaded.getvalue()
        _enc2  = _cd2.detect(_raw2[:20000])["encoding"] or "latin-1"
        _text2 = _raw2.decode(_enc2, errors="replace")
        _sep2  = ";" if _text2.split("\n")[0].count(";") > _text2.split("\n")[0].count(",") else ","
        _peek2 = pd.read_csv(io.StringIO(_text2), sep=_sep2, nrows=0)
        _peek2.columns = _peek2.columns.str.strip()
        filter_col_name = st.selectbox("Filter column", options=list(_peek2.columns))
        _full2 = pd.read_csv(io.StringIO(_text2), sep=_sep2, usecols=[filter_col_name])
        uniq   = sorted(_full2[filter_col_name].dropna().astype(str).unique())
        filter_vals = st.multiselect("Keep values", options=uniq, default=uniq)

    st.divider()
    st.subheader("📊 Select plots")
    plot_names = [p[0] for p in PLOTS]
    selected   = st.multiselect("Show plots", options=plot_names, default=[])

    st.divider()
    st.subheader("🔧 Custom bivariate plot")
    st.caption("Select two numeric columns to add a plot.")

    _custom_cols = []
    if uploaded:
        import chardet as _cdc
        _rawc  = uploaded.getvalue()
        _encc  = _cdc.detect(_rawc[:20000])["encoding"] or "latin-1"
        _textc = _rawc.decode(_encc, errors="replace")
        _sepc  = ";" if _textc.split("\n")[0].count(";") > _textc.split("\n")[0].count(",") else ","
        _dfc   = pd.read_csv(io.StringIO(_textc), sep=_sepc, nrows=5)
        _dfc.columns = _dfc.columns.str.strip()
        _custom_cols = [c for c in _dfc.columns
                        if pd.to_numeric(_dfc[c], errors="coerce").notna().any()]

    if _custom_cols:
        _NONE = "— select a column —"
        _opts = [_NONE] + _custom_cols
        _cx = st.selectbox("X axis", options=_opts, key="cust_x")
        _cy = st.selectbox("Y axis", options=_opts, key="cust_y")
        _cl, _cr = st.columns(2)
        _log_x = _cl.checkbox("log X", key="cust_logx")
        _log_y = _cr.checkbox("log Y", key="cust_logy")
        if _cx != _NONE and _cy != _NONE:
            st.session_state["custom_plot_active"] = True
            st.session_state["custom_plot_cfg"] = (_cx, _cy, _log_x, _log_y)
        if st.session_state.get("custom_plot_active"):
            if st.button("🗑 Remove custom plot", use_container_width=True):
                st.session_state["custom_plot_active"] = False
                st.session_state["custom_plot_cfg"] = None
                st.rerun()
    else:
        st.caption("Upload a CSV to enable custom plot.")

    st.divider()
    st.subheader("🗺️ Shapefile layers")
    st.caption("Select all shapefile files (.shp .dbf .shx .prj …) together.")

    _upkey = st.session_state["shp_uploader_key"]
    shp_uploads = st.file_uploader(
        "Shapefile components",
        type=["shp","dbf","shx","prj","cpg","qpj"],
        accept_multiple_files=True,
        key=f"shp_uploader_{_upkey}",
    )
    shp_zip = st.file_uploader(
        "Or upload as ZIP",
        type=["zip"],
        key=f"shp_zip_uploader_{_upkey}",
    )
    if st.button("🗑 Clear all shapefiles", use_container_width=True):
        st.session_state["shp_layers"] = []
        st.session_state["shp_colors"] = {}
        st.session_state["shp_opacity"] = {}
        st.session_state["shp_uploader_key"] += 1
        st.rerun()

# ── Guard ─────────────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown("""
    <div style="padding:60px 20px 0 20px">
    <h1>💧 Hydrochemistry Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈 Upload a hydrochemistry CSV in the sidebar to get started.")
    st.stop()

# ── Load & transform ──────────────────────────────────────────────────────────
df, COL = load_data(uploaded.getvalue(), uploaded.name)

if label_column and label_column in df.columns:
    df["Name"] = df[label_column].astype(str)

if filter_on and filter_col_name and filter_vals:
    col_map  = {c.strip().lower(): c for c in df.columns}
    resolved = filter_col_name if filter_col_name in df.columns \
               else col_map.get(filter_col_name.strip().lower(), filter_col_name)
    mask = df[resolved].astype(str).str.strip().isin([str(v).strip() for v in filter_vals])
    df   = df[mask].copy()
    st.sidebar.success(f"Filter: {len(df)} rows kept")

selected_name = st.session_state.get("selected_point", None)
has_coords    = df["_lon"].notna().any()

# ── Process newly uploaded shapefiles ─────────────────────────────────────────
def _process_shp_uploads(uploads, zip_file):
    added = False
    if uploads:
        # Group all uploaded files by their base name (without extension)
        # so that layer1.shp + layer1.dbf + layer2.shp + layer2.dbf are handled correctly
        from collections import defaultdict
        groups = defaultdict(dict)
        for f in uploads:
            base = f.name.rsplit(".", 1)[0]
            groups[base][f.name] = f.getvalue()

        existing = [n for n, _ in st.session_state["shp_layers"]]
        for layer_name, file_dict in groups.items():
            # Only process groups that actually contain a .shp file
            has_shp = any(fn.lower().endswith(".shp") for fn in file_dict)
            if not has_shp:
                continue
            if layer_name in existing:
                continue
            gdf = load_shapefile(file_dict)
            if gdf is not None:
                st.session_state["shp_layers"].append((layer_name, gdf))
                existing.append(layer_name)
                added = True

    if zip_file:
        zip_bytes = zip_file.getvalue()
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = os.path.join(tmp, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_bytes)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            for root, dirs, files in os.walk(tmp):
                for fname in files:
                    if fname.lower().endswith(".shp"):
                        layer_name = fname[:-4]
                        existing   = [n for n, _ in st.session_state["shp_layers"]]
                        if layer_name not in existing:
                            fdict = {}
                            for sibling in os.listdir(root):
                                spath = os.path.join(root, sibling)
                                if os.path.isfile(spath):
                                    with open(spath, "rb") as sf:
                                        fdict[sibling] = sf.read()
                            gdf = load_shapefile(fdict)
                            if gdf is not None:
                                st.session_state["shp_layers"].append((layer_name, gdf))
                                added = True
    return added

if _process_shp_uploads(shp_uploads, shp_zip):
    st.rerun()

shp_layers = st.session_state["shp_layers"]

# Assign default colour/opacity for any new layer
for i, (lname, _) in enumerate(shp_layers):
    if lname not in st.session_state["shp_colors"]:
        st.session_state["shp_colors"][lname] = _SHP_COLORS[i % len(_SHP_COLORS)]
    if lname not in st.session_state["shp_opacity"]:
        st.session_state["shp_opacity"][lname] = 0.5

# Show loaded layers + controls in sidebar
if shp_layers:
    with st.sidebar:
        st.markdown("**Loaded layers:**")
        for lname, gdf in shp_layers:
            st.markdown(
                f'<span style="font-size:0.85em;font-weight:600;">{lname}</span> '
                f'<span style="color:#888;font-size:0.78em;">({len(gdf)} features)</span>',
                unsafe_allow_html=True)
            ca, cb, cc = st.columns([2, 2, 1])
            new_color = ca.color_picker(
                "Colour", value=st.session_state["shp_colors"][lname],
                key=f"clr_{lname}")
            new_opacity = cb.slider(
                "Opacity", 0.0, 1.0,
                value=st.session_state["shp_opacity"][lname],
                step=0.05, key=f"opa_{lname}")
            st.session_state["shp_colors"][lname] = new_color
            st.session_state["shp_opacity"][lname] = new_opacity
            if cc.button("✕", key=f"rm_{lname}"):
                st.session_state["shp_layers"] = [
                    (n, g) for (n, g) in st.session_state["shp_layers"] if n != lname
                ]
                st.session_state["shp_colors"].pop(lname, None)
                st.session_state["shp_opacity"].pop(lname, None)
                st.session_state["shp_uploader_key"] += 1
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  STICKY MAP PANEL
# ══════════════════════════════════════════════════════════════════════════════
# Open the sticky wrapper div
st.markdown('<div id="map-sticky-wrapper">', unsafe_allow_html=True)

# Title + metrics inline
n_samples  = len(df)
n_clusters = df["Cluster"].nunique()
n_aquifers = df["Aquifer"].nunique()
n_cols     = sum(1 for v in COL.values() if v)
banner     = f'''
<div style="display:flex; align-items:center; justify-content:space-between; padding:2px 4px 0 4px">
  <span style="font-size:1.35em; font-weight:700; color:#185FA5;">💧 Hydrochemistry Dashboard</span>
  <span style="display:flex; gap:12px;">
    <span class="metric-box"><div class="val">{n_samples}</div><div class="lbl">Samples</div></span>
    <span class="metric-box"><div class="val">{n_clusters}</div><div class="lbl">Clusters</div></span>
    <span class="metric-box"><div class="val">{n_aquifers}</div><div class="lbl">Aquifers</div></span>
    <span class="metric-box"><div class="val">{n_cols}</div><div class="lbl">Columns matched</div></span>
  </span>
</div>
'''
st.markdown(banner, unsafe_allow_html=True)

if selected_name:
    st.markdown(
        f'<div class="selected-banner">📍 Selected: {selected_name} ' +
        '<span style="font-weight:400;font-size:0.9em;color:#999;">(click another point or use Clear selection to deselect)</span></div>',
        unsafe_allow_html=True)

if has_coords:
    map_fig = build_map(df, symbology, show_labels, selected_name, shp_layers)
    if map_fig:
        map_fig.update_layout(height=520)
        map_event = st.plotly_chart(
            map_fig,
            use_container_width=True,
            key="map_chart",
            on_select="rerun",
            selection_mode="points",
        )
        if map_event and hasattr(map_event, "selection") and map_event.selection:
            pts = map_event.selection.get("points", [])
            if pts:
                clicked = pts[0].get("customdata")
                if clicked and clicked != selected_name:
                    st.session_state["selected_point"] = clicked
                    st.rerun()
else:
    st.info("ℹ️ No Lambert 93 coordinates found (expected: 'X L93 m' / 'Y L93 M').")

# Close sticky wrapper
st.markdown("</div>", unsafe_allow_html=True)

# ── Spacer so content starts below the sticky panel ───────────────────────────
# We inject a JS snippet to measure the sticky panel height and push content down
st.components.v1.html("""
<script>
(function() {
  function adjustPadding() {
    const wrapper = window.parent.document.getElementById("map-sticky-wrapper");
    const mainBlock = window.parent.document.querySelector(".main .block-container");
    if (wrapper && mainBlock) {
      const h = wrapper.getBoundingClientRect().height;
      mainBlock.style.paddingTop = (h + 8) + "px";
    }
  }
  adjustPadding();
  setTimeout(adjustPadding, 600);
  window.addEventListener("resize", adjustPadding);
})();
</script>
""", height=0)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS  — 1 per row
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📊 Hydrochemical plots")
selected_plots = [p for p in PLOTS if p[0] in selected]

if not selected_plots:
    st.warning("No plots selected. Choose some in the sidebar.")
else:
    for idx, (name, builder) in enumerate(selected_plots):
        fig = builder(df, COL, symbology, show_labels, selected_name)
        if fig is None:
            with st.expander(f"⚠️ {name} — skipped"):
                st.caption("Required columns not found.")
            continue

        fig.update_layout(height=520)
        plot_event = st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"plot_{idx}",
            on_select="rerun",
            selection_mode="points",
        )
        if plot_event and hasattr(plot_event, "selection") and plot_event.selection:
            pts = plot_event.selection.get("points", [])
            if pts:
                clicked = pts[0].get("customdata")
                if clicked and clicked != selected_name:
                    st.session_state["selected_point"] = clicked
                    st.rerun()

# ── Custom bivariate plot ─────────────────────────────────────────────────────
if st.session_state.get("custom_plot_active") and st.session_state.get("custom_plot_cfg"):
    cx, cy, log_x, log_y = st.session_state["custom_plot_cfg"]
    st.markdown("### 🔧 Custom bivariate plot")
    if cx not in df.columns or cy not in df.columns:
        st.warning(f"Column '{cx}' or '{cy}' not found in the data.")
    else:
        _xv = pd.to_numeric(df[cx], errors="coerce")
        _yv = pd.to_numeric(df[cy], errors="coerce")
        _mask = _xv.notna() & _yv.notna()
        if log_x: _mask &= (_xv > 0)
        if log_y: _mask &= (_yv > 0)
        _d = df[_mask].copy()
        if _d.empty:
            st.warning(f"No valid numeric data for {cx} vs {cy}.")
        else:
            _xlabel = f"log({cx})" if log_x else cx
            _ylabel = f"log({cy})" if log_y else cy
            _fig = base_fig(f"{cx} vs {cy}", _xlabel, _ylabel)
            for _t in make_traces(_d, _xv[_mask], _yv[_mask], symbology, show_labels, selected_name):
                _fig.add_trace(_t)
            if log_x: _fig.update_xaxes(type="log", dtick=1, tickformat=".3~g", minor=dict(ticks="inside", ticklen=3, showgrid=True, gridcolor="#ececec"))
            if log_y: _fig.update_yaxes(type="log", dtick=1, tickformat=".3~g", minor=dict(ticks="inside", ticklen=3, showgrid=True, gridcolor="#ececec"))
            _fig.update_layout(height=520)
            _cevt = st.plotly_chart(
                _fig,
                use_container_width=True,
                key="custom_plot",
                on_select="rerun",
                selection_mode="points",
            )
            if _cevt and hasattr(_cevt, "selection") and _cevt.selection:
                _pts = _cevt.selection.get("points", [])
                if _pts:
                    _clicked = _pts[0].get("customdata")
                    if _clicked and _clicked != selected_name:
                        st.session_state["selected_point"] = _clicked
                        st.rerun()
