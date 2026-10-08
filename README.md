# Fieldwise · Smart Irrigation Prototype

A software-only irrigation decision-support prototype: one page dashboard, local CSV upload, and a Python feature-preparation pipeline. It does not connect to valves, pumps, controllers, or real sensors. Dashboard values are illustrative until real farm records are loaded.

## Open the dashboard

Open `index.html` in a browser. The included values are an illustrative Central Valley sample, not live readings. The crop selector recalculates the explanatory recommendation. `Refresh weather` requests NASA POWER daily weather for the coordinates shown in the location control. Browser access to the NASA service depends on network/CORS availability. The CSV uploader parses files locally and uses the latest row with recognized fields to update available indicators; it does not upload the file.

For best browser compatibility, serve this folder from a local static web server. No frontend build step is required.

## Data sources and limits

* **NASA POWER Daily API, Agroclimatology community:** fetched by the pipeline for a point location and date range. Fields include daily mean/min/max temperature, humidity, wind, precipitation, and solar radiation. See [NASA POWER daily API documentation](https://power.larc.nasa.gov/docs/services/api/temporal/daily/).
* **Kaggle precision-agriculture dataset:** download the CSV manually from [the provided Kaggle dataset page](https://www.kaggle.com/datasets/colabsss/precision-agriculture-monitoring-dataset) and pass it to the pipeline. This environment could read the page title but did not expose its files. The [Kaggle smart-crop notebook](https://www.kaggle.com/code/lukhilaksh/smart-crop-98-beats-accuracy/notebook) can inform feature/label exploration; its reported accuracy is not treated as verified or reproduced. The provided [BigQuery AI hackathon writeup URL](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/placeholder-title) appears to be a writeup page, not a data asset.
* **USDA NRCS SSURGO:** U.S.-only official surveyed soil profiles can be downloaded by area via [Web Soil Survey / SSURGO](https://www.nrcs.usda.gov/resources/data-and-reports/soil-survey-geographic-database-ssurgo). Export/prepare the fields of interest (for example pH, sand, clay, organic carbon) as a CSV with `location_id` matching the field records, or a one-row selected field profile. SSURGO does not provide global coverage.

The actual Kaggle CSVs were not bundled because Kaggle access/authentication was unavailable from this workspace. No Kaggle data is silently fabricated. The dashboard's built-in records are demo-only. Source attribution and units should be confirmed from each downloaded dataset's own metadata before combining it; do not assume similar column names mean identical measurement methods.

## Unified frame and features

`pipeline/prepare_data.py` normalizes common column aliases, parses dates, joins dated records on shared `date` and `location_id` keys, attaches static soil rows by `location_id` (or broadcasts a one-row profile), and writes `output/merged_feature_frame.csv` plus `output/pipeline_summary.json`.

Engineered columns include:

* `et0_mm`: temperature-based Hargreaves-Samani estimate when daily mean/min/max temperatures are available (mean ±5°C fallback when extrema are absent).
* `crop_et_mm`: estimated reference ET multiplied by the selected crop coefficient (`--kc`).
* `effective_rain_mm`, `net_water_balance_mm`, and `soil_water_deficit_mm` (when soil moisture is present).
* `proxy_irrigation_need_mm` and `irrigation_recommended_proxy`: transparent decision-support heuristics, **not observed irrigation labels or a validated model output**.

Example (after downloading the Kaggle CSV locally):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python pipeline/prepare_data.py --input .\data\precision_agriculture.csv --input .\data\field_soil.csv --latitude 36.77 --longitude -119.42 --start 2025-01-01 --end 2025-12-31 --crop tomato --kc 1.15
```

Repeat `--input` for sensor, crop, and soil CSVs. Common column names are mapped by the adapter (see `ALIASES` in the script). Use consistent units: soil moisture as percent or fraction, temperatures in °C, rainfall in mm, wind in m/s, soil fractions as percent. Set a meaningful shared `location_id` on records when available. Keep soil profiles static or dated observations clearly distinguished.

## Optional ML baseline

When at least 30 rows include an **observed** `irrigation_needed` or `irrigation_need_mm` label, the script trains an optional Random Forest baseline and reports a held-out metric in the summary. It never trains on its own proxy target. A random split is only a smoke-level prototype diagnostic: before making operational claims, validate across time, fields, crops, and irrigation systems; inspect class balance, leakage, calibration, and agronomic constraints. The dashboard currently uses interpretable rules, not the exported model artifact.

## 10-day delivery path

1. Confirm location, crops, irrigation method, and source licenses/terms; download Kaggle data and verify its schema.
2. Map each source's variable meanings, units, dates, and field identifiers; prepare the first merged frame.
3. Add local soil survey and sensor history; check missingness, duplicates, and location/date joins.
4. Define labels from observed irrigation actions or validated agronomic targets; split by time/farm.
5. Evaluate a baseline against simple ET/rule heuristics; review false recommendations with an agronomist.
6. Connect the selected feature frame to the dashboard and agree on uncertainty/fallback behavior.
7. Validate a second crop/field and check weather-source gaps.
8. Review dashboard usability and data provenance with intended users.
9. Document model limits, thresholds, and deployment/security requirements.
10. Demo with a fixed validated dataset and a clear decision-support disclaimer.
