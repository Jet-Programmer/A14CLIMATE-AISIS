"""Merge local agriculture CSVs with NASA POWER weather into model-ready features.

Example:
  python pipeline/prepare_data.py --input kaggle_precision.csv --input crop_observations.csv \
    --latitude 36.77 --longitude -119.42 --start 2024-01-01 --end 2024-12-31

The Kaggle files must be downloaded by the user. NASA POWER is fetched from its
public API; the merged frame and data provenance are written to ./output.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

ALIASES = {
    "date": ("date", "datetime", "timestamp", "time", "observation_date", "day"),
    "location_id": ("location_id", "field_id", "farm_id", "plot_id", "station_id", "site_id"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon", "lng"),
    "crop": ("crop", "crop_type", "crop_name"),
    "growth_stage": ("growth_stage", "phenological_stage", "crop_stage", "stage"),
    "soil_moisture_pct": ("soil_moisture_pct", "soil_moisture", "moisture_pct", "vwc", "soil_vwc", "sm"),
    "temperature_c": ("temperature_c", "temperature", "temp", "t2m"),
    "humidity_pct": ("humidity_pct", "humidity", "relative_humidity", "rh2m"),
    "rainfall_mm": ("rainfall_mm", "rainfall", "precipitation", "rain", "prectotcorr"),
    "wind_speed_m_s": ("wind_speed_m_s", "wind_speed", "wind", "ws2m"),
    "soil_ph": ("soil_ph", "ph"),
    "sand_pct": ("sand_pct", "sand", "sand_content"),
    "clay_pct": ("clay_pct", "clay", "clay_content"),
    "organic_carbon_pct": ("organic_carbon_pct", "organic_carbon", "soc"),
    "irrigation_needed": ("irrigation_needed", "needs_irrigation", "irrigate"),
    "irrigation_need_mm": ("irrigation_need_mm", "irrigation_depth_mm", "water_applied_mm", "target_irrigation_mm"),
}
KEY_CANDIDATES = ("date", "location_id")


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [slug(str(c)) for c in frame.columns]
    for canonical, aliases in ALIASES.items():
        if canonical not in frame:
            match = next((a for a in aliases if a in frame), None)
            if match:
                frame.rename(columns={match: canonical}, inplace=True)
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    if "soil_moisture_pct" in frame:
        # Convert fractional volumetric water content to percent only when all
        # plausible values are fractions. Values already expressed as percent stay unchanged.
        numeric = pd.to_numeric(frame["soil_moisture_pct"], errors="coerce")
        if numeric.notna().any() and numeric.dropna().quantile(.95) <= 1.0:
            frame["soil_moisture_pct"] = numeric * 100
    return frame


def fetch_nasa_power(latitude: float, longitude: float, start: str, end: str) -> pd.DataFrame:
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {"parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
              "community": "AG", "latitude": latitude, "longitude": longitude,
              "start": pd.Timestamp(start).strftime("%Y%m%d"), "end": pd.Timestamp(end).strftime("%Y%m%d"), "format": "JSON"}
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    body = response.json()
    params_data = body.get("properties", {}).get("parameter", {})
    if not params_data or "T2M" not in params_data:
        raise ValueError("NASA POWER returned no weather parameters for this period/location")
    out = pd.DataFrame(params_data).rename(columns={"T2M": "temperature_c", "T2M_MAX": "temperature_max_c",
        "T2M_MIN": "temperature_min_c", "RH2M": "humidity_pct", "WS2M": "wind_speed_m_s",
        "PRECTOTCORR": "rainfall_mm", "ALLSKY_SFC_SW_DWN": "solar_radiation_mj_m2"})
    out.index.name = "date"
    out = out.reset_index()
    for column in out.columns.difference(["date"]):
        out[column] = pd.to_numeric(out[column], errors="coerce").replace(-999, pd.NA)
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    out["latitude"], out["longitude"] = latitude, longitude
    out["location_id"] = f"{latitude:.5f},{longitude:.5f}"
    out["data_source"] = "NASA POWER Daily API (AG)"
    return out


def read_sources(paths: Iterable[Path]) -> list[pd.DataFrame]:
    frames = []
    for path in paths:
        frame = normalize_columns(pd.read_csv(path))
        if frame.empty:
            continue
        frame["data_source"] = path.name
        frames.append(frame)
    return frames


def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-merge by shared date/location keys; cross-join single-row static profiles."""
    if not frames:
        raise ValueError("No input frames to merge")
    dated = [f.copy() for f in frames if "date" in f]
    static = [f.copy() for f in frames if "date" not in f]
    if not dated:
        # A row-wise frame is still useful for static crop/soil profiles.
        merged = static.pop(0)
        for right in static:
            keys = [k for k in ("location_id",) if k in merged and k in right]
            merged = merged.merge(right, on=keys or None, how="outer", suffixes=("", "_source"))
        return merged
    merged = dated.pop(0)
    for right in dated:
        keys = [k for k in KEY_CANDIDATES if k in merged and k in right]
        if not keys:
            keys = ["date"] if "date" in merged and "date" in right else []
        merged = merged.merge(right, on=keys or None, how="outer", suffixes=("", "_source"))
    for right in static:
        keys = ["location_id"] if "location_id" in merged and "location_id" in right else []
        # Static soil-profile tables commonly have one selected field record.
        if not keys and len(right) == 1:
            for column in right.columns:
                if column not in merged and column not in {"data_source"}:
                    merged[column] = right.iloc[0][column]
            continue
        if keys:
            merged = merged.merge(right, on=keys, how="left", suffixes=("", "_soil"))
    return merged


def engineer_features(frame: pd.DataFrame, default_crop: str, kc: float, root_zone_mm: float) -> pd.DataFrame:
    out = frame.copy()
    out["crop"] = out.get("crop", pd.Series(default_crop, index=out.index)).fillna(default_crop)
    for col in ("temperature_c", "temperature_max_c", "temperature_min_c", "humidity_pct", "rainfall_mm", "soil_moisture_pct", "wind_speed_m_s"):
        if col not in out:
            out[col] = pd.NA
    for col in ("humidity_pct", "soil_moisture_pct", "wind_speed_m_s", "soil_ph", "sand_pct", "clay_pct", "organic_carbon_pct"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["rainfall_mm"] = pd.to_numeric(out["rainfall_mm"], errors="coerce").fillna(0).clip(lower=0)
    out["temperature_c"] = pd.to_numeric(out["temperature_c"], errors="coerce")
    # Use supplied Tmax/Tmin where available; approximate with Tmean ± 5°C otherwise.
    tmax = pd.to_numeric(out["temperature_max_c"], errors="coerce").fillna(out["temperature_c"] + 5)
    tmin = pd.to_numeric(out["temperature_min_c"], errors="coerce").fillna(out["temperature_c"] - 5)
    out["et0_mm"] = (0.0023 * (out["temperature_c"] + 17.8) * (tmax - tmin).clip(lower=0).pow(.5)).clip(lower=0)
    out["crop_coefficient_kc"] = kc
    out["crop_et_mm"] = out["et0_mm"] * out["crop_coefficient_kc"]
    out["effective_rain_mm"] = out["rainfall_mm"] * .8
    moisture = pd.to_numeric(out["soil_moisture_pct"], errors="coerce")
    out["soil_water_deficit_mm"] = ((100 - moisture) / 100 * root_zone_mm).clip(lower=0)
    out["net_water_balance_mm"] = out["effective_rain_mm"] - out["crop_et_mm"]
    # This is a transparent agronomic proxy, not observed irrigation demand.
    out["proxy_irrigation_need_mm"] = (out["crop_et_mm"] - out["effective_rain_mm"]).clip(lower=0)
    out["irrigation_recommended_proxy"] = (out["proxy_irrigation_need_mm"] >= 3) | (moisture < 30)
    out["feature_frame_version"] = "1.0"
    return out


def train_if_labeled(frame: pd.DataFrame, out_dir: Path) -> dict:
    """Train an optional baseline classifier only when real labels are available."""
    label = "irrigation_needed" if "irrigation_needed" in frame else None
    if not label and "irrigation_need_mm" in frame:
        label = "irrigation_need_mm"
    if not label:
        return {"status": "skipped", "reason": "No observed irrigation outcome/label column. Proxy targets are not used as ground truth."}
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, mean_absolute_error
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder
        import joblib
    except ImportError:
        return {"status": "skipped", "reason": "Install scikit-learn and joblib to train an optional baseline."}
    candidate = [c for c in ("soil_moisture_pct", "temperature_c", "humidity_pct", "rainfall_mm", "wind_speed_m_s", "et0_mm", "crop_et_mm", "crop", "growth_stage", "soil_ph", "sand_pct", "clay_pct") if c in frame]
    valid = frame.dropna(subset=[label]).copy()
    if len(valid) < 30:
        return {"status": "skipped", "reason": f"Only {len(valid)} labeled rows; at least 30 are required for this demo baseline."}
    cats = [c for c in candidate if valid[c].dtype == "object"]
    nums = [c for c in candidate if c not in cats]
    transformers = []
    if nums: transformers.append(("numeric", make_pipeline(SimpleImputer(strategy="median")), nums))
    if cats: transformers.append(("category", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cats))
    pre = ColumnTransformer(transformers)
    is_class = label == "irrigation_needed"
    model = RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42) if is_class else RandomForestRegressor(n_estimators=250, random_state=42)
    pipeline = make_pipeline(pre, model)
    # Stable deterministic split; callers should validate by time/farm before deployment.
    cut = int(len(valid) * .8)
    if cut < 1 or len(valid) - cut < 1:
        return {"status": "skipped", "reason": "Not enough labeled records for a train/validation split."}
    pipeline.fit(valid[candidate].iloc[:cut], valid[label].iloc[:cut])
    pred = pipeline.predict(valid[candidate].iloc[cut:])
    metric = accuracy_score(valid[label].iloc[cut:], pred) if is_class else mean_absolute_error(valid[label].iloc[cut:], pred)
    filename = "irrigation_classifier.joblib" if is_class else "irrigation_depth_regressor.joblib"
    joblib.dump(pipeline, out_dir / filename)
    return {"status": "trained", "label": label, "train_rows": cut, "validation_rows": len(valid)-cut,
            "metric": "accuracy" if is_class else "mae_mm", "value": float(metric), "model": filename,
            "caveat": "Single random split is a prototype diagnostic; validate chronologically and by unseen farms before use."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, default=[], help="Local Kaggle/sensor/soil CSV; repeat for multiple files")
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--crop", default="tomato")
    parser.add_argument("--kc", type=float, default=1.15, help="Crop coefficient for selected crop/stage")
    parser.add_argument("--root-zone-mm", type=float, default=600)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    for path in args.input:
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weather = fetch_nasa_power(args.latitude, args.longitude, args.start, args.end)
    local = read_sources(args.input)
    # Exclude a local file's date-less metadata from dated join surprises; static one-row soil data broadcasts.
    merged = merge_frames([weather, *local])
    featured = engineer_features(merged, args.crop, args.kc, args.root_zone_mm)
    output = args.output_dir / "merged_feature_frame.csv"
    featured.to_csv(output, index=False)
    model_result = train_if_labeled(featured, args.output_dir)
    summary = {"rows": int(len(featured)), "columns": list(featured.columns), "output": str(output),
               "sources": ["NASA POWER Daily API (AG)", *[p.name for p in args.input]],
               "proxy_target_warning": "proxy_irrigation_need_mm is calculated from weather balance and is not an observed label",
               "model": model_result}
    (args.output_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
