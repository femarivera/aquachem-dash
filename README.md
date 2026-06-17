# AQUACHEM DASHBOARD

An interactive hydrochemical data dashboard built with Streamlit and Plotly. Upload a CSV file containing data point coordinates and chemical data, explore 26 predefined geochemical plots, visualize sample locations on an interactive map, and build custom bivariate plots.

**Live app → [share.streamlit.io](https://aquachem-dash.streamlit.app/)**

---

## Features

**26 predefined hydrochemical plots**, including:
- Chloro-Alkaline Indices (CAI)
- Na-normalised and unnormalised ionic ratios
- Cation exchange balance
- pCO₂, SI Calcite, and carbonate equilibrium diagrams
- ¹⁴C / δ¹³C diagnostic areas, mixing lines, and Pearson model
- NETPATH vs Pearson radiocarbon correction comparisons
- Ion–ion biplots (Na, Ca, Mg, Cl, SO₄, HCO₃)

**Interactive map panel**
- Automatic coordinate detection and Lambert 93 → WGS 84 reprojection
- Optional shapefile overlay (upload `.shp`, `.dbf`, `.shx`, `.prj` or a `.zip`)
- Click a point on the map to highlight it across all plots

**Cross-plot highlighting**
- Click any data point in any plot to highlight that sample everywhere
- Gold marker style makes selected samples instantly visible
- Useful to track data points across flowpaths and hydrochemical plots

**Custom bivariate plot**
- Select any two numeric columns from your data as X and Y axes
- Optional log scale on either axis, with clean decade ticks and minor gridlines
- Updates live as you change the selection

**Flexible symbology**
- Colour by Aquifer (PlioQ, Miocene, Eocene, Oligocene) or by Cluster
- Consistent palette and marker shapes across all plots and the map

---

## Input data format

Aquachem Dashboard expects a **CSV file**. The column names are detected automatically using fuzzy matching. Key columns the app looks for:

| Data | Example column names |
|---|---|
| Coordinates | `X`, `Y`, `Lambert_X`, `Longitude`, `Latitude` |
| Major ions | `Na (ppm)`, `Ca (ppm)`, `Cl (ppm)`, `HCO3 (ppm)`, … |
| Isotopes | `d18O`, `D2H`, `14C (pmc)`, `d13C` |
| Sample label | `Name`, `Sample`, `Borehole`, `Well` |
| Geochemical indices | `SI_Calcite` |

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `plotly` | Interactive plots |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `pyproj` | Coordinate reprojection (Lambert 93 → WGS 84) |
| `chardet` | Automatic CSV encoding detection |

## License

MIT
