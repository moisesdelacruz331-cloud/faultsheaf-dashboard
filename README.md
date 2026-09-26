# FaultSheaf Hazard Explorer

A Streamlit demonstration app built from the FaultSheaf research study
(Non-Euclidean Cellular Networks for Philippine Seismic Hazard Forecasting).

## What this is

This app is a **results explorer**, not a re-implementation of the research.
Every map and every number shown was produced by the study's own locked,
leakage-free pipeline (`pipeline2.py`, not included here) and exported once
into the `faultsheaf_app/` data folder. The app loads those files directly —
nothing is recomputed live, so what you see here is guaranteed to match the
paper exactly.

## Contents

- `app.py` — the Streamlit application (6 pages/sections)
- `faultsheaf_app/hazard_grids.npz` — the four models' real, precomputed
  probability grids (Uniform Null, Smoothed Seismicity, Isotropic Graph
  Diffusion, Sheaf-Diffusion), at 180×120 resolution (21,600 cells)
- `faultsheaf_app/test_target_events.csv` — the real 2,892 M≥3.0 PHIVOLCS
  earthquake events from the Jan 1 – Aug 1, 2026 prospective test window,
  used only for map overlay and never for any live computation
- `faultsheaf_app/study_summary.json` — every statistic quoted in the app
  (Information Gain, AUC, hit rates, bootstrap CIs, ablation results, the
  shuffle-test result, data provenance), copied directly from the study's
  own results tables
- `requirements.txt` — Python package dependencies

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (typically http://localhost:8501).

## App sections

1. **Study Overview** — headline results and the three research questions
2. **Interactive Hazard Map** — switchable model layers, real event overlay,
   click-to-query any grid cell's probability
3. **Model Comparison** — full results table, validation-to-test generalization
4. **Statistical Significance** — bootstrap/t-test results, with the honest
   event-independence caveat stated explicitly
5. **Ablation & Robustness Test** — the graph-diffusion vs. anisotropy
   decomposition, and the shuffled-orientation test reported exactly as found
   (including the result that went against the original hypothesis)
6. **About the Data** — real magnitude distribution, catalog completeness
   caveats (missing 2020–2022), and the computational benchmark, all stated
   with the same precision as the paper

## A note on scope

This prototype demonstrates the validated research model's output. It is not
a live, continuously-updated forecasting service — the underlying model would
need to be retrained and re-validated on new data before being used
operationally, following the same leakage-free protocol described in the
paper.
