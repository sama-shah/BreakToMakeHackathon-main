
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from io import StringIO
from typing import List, Optional

# Instructions
# streamlit run app.py

# ----------------------
# App Config
# ----------------------
st.set_page_config(page_title="Cycle Phase & Ovulation Detector (Oura + BBT)", layout="wide")
st.title("Cycle Phase & Ovulation Detector")
st.caption("Drag-and-drop your Oura CSV or stream a 10k NTC sensor (Arduino) to compare simulated BBT and Oura temp-trend side-by-side.")

with st.sidebar:
    st.header("Modes & Settings")
    mode = st.radio("Choose Mode", ["Oura CSV Upload", "Sensor Stream (BBT) + Simulated Data"])
    st.markdown("---")
    st.subheader("Detection Parameters")
    RISE_MIN = st.slider("Sustained Rise Threshold (°C)", 0.1, 0.6, 0.25, 0.01)
    RISE_DAYS = st.slider("Rise Must Persist (days)", 2, 5, 3, 1)
    SMOOTH_WIN = st.slider("Smoothing Window (days)", 3, 9, 5, 1)
    st.markdown("---")
    st.subheader("Optional Menses Start Dates")
    menses_text = st.text_area("Comma-separated YYYY-MM-DD (e.g. 2025-08-04, 2025-09-01)", "")
    if menses_text.strip():
        try:
            menses_starts = [pd.to_datetime(x.strip()).date() for x in menses_text.split(",") if x.strip()]
        except Exception:
            st.warning("Could not parse some dates. Check format YYYY-MM-DD.")
            menses_starts = []
    else:
        menses_starts = []

# ----------------------
# Core detection code
# ----------------------

def _extract_temp_trend_from_any(df: pd.DataFrame, smooth_win: int) -> pd.Series:
    """
    Return a Series for temperature trend/deviation in °C if possible.
    Supports:
      - Direct columns: temp_trend_deviation, temperature_trend_deviation, temp_trend
      - Deviation columns: temperature_deviation, temp_deviation (smoothed)
      - Dotted/underscored paths like: readiness.contributors.temperature_trend_deviation
      - JSON-like readiness column with a dict containing contributors.temperature_trend_deviation
    """
    # 1) Direct/simple candidates
    direct_candidates = [
        "temp_trend_deviation", "temperature_trend_deviation", "temp_trend",
        "temperature_deviation", "temp_deviation"
    ]
    for c in df.columns:
        lc = c.lower().strip()
        if lc in direct_candidates:
            s = pd.to_numeric(df[c], errors="coerce")
            if lc in ["temperature_deviation", "temp_deviation"]:
                return s.rolling(smooth_win, min_periods=1, center=True).mean()
            return s

    # 2) Columns that *contain* the key string (dotted/underscored export styles)
    for c in df.columns:
        lc = c.lower().strip()
        if ("temperature_trend_deviation" in lc or "temp_trend_deviation" in lc or "temperature_deviation" in lc):
            return pd.to_numeric(df[c], errors="coerce")

    # 3) Parse readiness column if present (JSON/dict or stringified JSON)
    #    Look for readiness['contributors']['temperature_trend_deviation']
    possible_readiness_cols = [c for c in df.columns if c.lower().strip() in ["readiness", "readiness_json", "readiness_data"]]
    for rc in possible_readiness_cols:
        try:
            # If column already expanded (dict-like), try dict access row-wise
            if isinstance(df[rc].iloc[0], dict):
                contrib = df[rc].apply(lambda x: x.get("contributors", {}) if isinstance(x, dict) else {})
                series = contrib.apply(lambda d: d.get("temperature_trend_deviation", np.nan))
                if series.notna().sum() > 0:
                    return pd.to_numeric(series, errors="coerce")
            else:
                # Try JSON parsing per row
                import json
                parsed = df[rc].apply(
                    lambda s: json.loads(s) if isinstance(s, str) and s.strip().startswith("{") else {}
                )
                contrib = parsed.apply(lambda x: x.get("contributors", {}) if isinstance(x, dict) else {})
                series = contrib.apply(lambda d: d.get("temperature_trend_deviation", np.nan))
                if series.notna().sum() > 0:
                    return pd.to_numeric(series, errors="coerce")
        except Exception:
            pass

    # 4) Nothing found
    raise ValueError("Could not find a temperature trend/deviation column. "
                     "Expected one of: temp_trend_deviation / temperature_trend_deviation "
                     "or a nested readiness.contributors.temperature_trend_deviation.")

def normalize_and_choose_temp(df: pd.DataFrame, cols, smooth_win: int) -> pd.DataFrame:
    # bring a 'date' column into a standard name
    date_candidates = ["date", "summary_date", "day"]
    date_col = None
    for c in df.columns:
        if c.lower().strip() in date_candidates:
            date_col = c
            break
    if date_col is None:
        raise ValueError("CSV must include a date/summary_date/day column.")
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").drop_duplicates("date")

    # temperature series extraction
    temp = _extract_temp_trend_from_any(df, smooth_win)

    # store both normalized and C-scale
    df["temp_signal_c"] = temp
    std = float(temp.std(ddof=0)) if float(temp.std(ddof=0)) > 0 else 1.0
    df["temp_signal"] = (temp - float(temp.mean())) / std

    # optional helpers auto-map (keep original names if present)
    helpers_map = {
        "rhr": ["resting_heart_rate", "rhr", "average_resting_heart_rate"],
        "hrv": ["hrv", "rmssd"],
        "resp_rate": ["respiratory_rate", "breath_average"]
    }
    for key, cands in helpers_map.items():
        for c in df.columns:
            if c.lower().strip() in cands:
                df = df.rename(columns={c: key})
                df[key] = pd.to_numeric(df[key], errors="coerce").interpolate(limit_direction="both")
                break
    return df

# Column mapping is now only used for date helpers; temp extraction is automatic.
CANDIDATE_COLS = {
    "date": ["date", "summary_date", "day"],
}

# ----------------------
# UI: Oura CSV Mode
# ----------------------
def oura_csv_ui():
    st.subheader("Upload Oura CSV")
    st.write("Your file should include a **date** column and at least one of **temp_trend_deviation** or **temperature_deviation**. Other helpful columns: resting_heart_rate, hrv (RMSSD), respiratory_rate.")
    upl = st.file_uploader("Drop CSV here", type=["csv"])

    if not upl:
        st.info("Waiting for CSV… You can also download a sample template from the sidebar README once the app is running locally.")
        return

    try:
        df = pd.read_csv(upl)
        df_proc = normalize_and_choose_temp(df, CANDIDATE_COLS, SMOOTH_WIN)
    except Exception as e:
        st.error(f"Failed to read/process CSV: {e}")
        return

    # If no menses provided, try inference (can be toggled later if needed)
    starts = menses_starts if len(menses_starts) > 0 else infer_menses_starts(df_proc)

    labeled, cycles = label_phases(df_proc, starts, RISE_MIN, RISE_DAYS)

    # Show results
    st.success("Processed!")
    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown("#### Temperature & Phase Timeline")
        # simple line chart
        chart_df = labeled[["date", "temp_signal_c", "phase"]].copy()
        chart_df["date"] = pd.to_datetime(chart_df["date"])
        st.line_chart(chart_df.set_index("date")[["temp_signal_c"]], height=280)
        st.caption("Hover to see values. Use the table below for phase labels.")

    with c2:
        st.markdown("#### Cycle Summary")
        if cycles:
            for cyc in cycles:
                st.write(f"- **Cycle start:** {cyc['cycle_start']}  \n"
                         f"  **Ovulation:** {cyc.get('ovulation', 'not found')}  \n"
                         f"  **Confidence:** {cyc.get('confidence', '—')}")
        else:
            st.write("No cycles summarized.")

    st.markdown("#### Labeled Data")
    st.dataframe(labeled, use_container_width=True, hide_index=True)

    # Download
    csv = labeled.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download labeled CSV", csv, file_name="oura_with_phases.csv", mime="text/csv")

# ----------------------
# UI: Sensor Stream + Sim
# ----------------------
def simulate_daily_bbt(days=60, baseline=0.0, ov_day=28, rise=0.35, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    temps = []
    for d in range(days):
        base = baseline + 0.02*np.sin(2*np.pi*d/27.0)  # gentle infradian-ish wobble
        bump = rise if d >= ov_day else 0.0
        temps.append(base + bump + rng.normal(0, noise))
    dates = [datetime.today().date() - timedelta(days=days - 1 - i) for i in range(days)]
    return pd.DataFrame({"date": dates, "temp_signal_c": temps})

def sensor_stream_ui():
    st.subheader("Sensor Stream (BBT) + Simulated Comparison")
    st.write("For hackathon demos without a full-night dataset, use the **Simulated BBT** stream and compare against Oura's **temp trend** from your CSV.")
    st.markdown("**Tip:** In a real build, wire your Arduino's serial output as `date,temperature_c` lines and replace the simulation block below with `pyserial` reads.")

    # Left: simulated BBT; Right: optional Oura CSV to overlay
    left, right = st.columns(2)

    with left:
        st.markdown("##### Simulated BBT")
        days = st.slider("Simulated days", 30, 120, 60, 1)
        ov_d = st.slider("Simulated ovulation day index", 10, days-10, min(28, days-10), 1)
        rise = st.slider("Simulated luteal rise (°C)", 0.15, 0.60, 0.35, 0.01)
        noise = st.slider("Noise (°C std)", 0.01, 0.15, 0.05, 0.01)
        sim = simulate_daily_bbt(days=days, ov_day=ov_d, rise=rise, noise=noise)
        st.line_chart(sim.set_index(pd.to_datetime(sim["date"]))[["temp_signal_c"]], height=260)
        st.caption("Simulated daily BBT-like series with ovulation-linked rise.")

    with right:
        st.markdown("##### (Optional) Oura CSV to Overlay")
        upl2 = st.file_uploader("Drop Oura CSV for overlay", type=["csv"], key="overlay")
        overlay_df = None
        if upl2:
            try:
                raw = pd.read_csv(upl2)
                overlay_df = normalize_and_choose_temp(raw, CANDIDATE_COLS, SMOOTH_WIN)
                overlay_df = overlay_df[["date", "temp_signal_c"]].copy()
                overlay_df["date"] = pd.to_datetime(overlay_df["date"])
                st.line_chart(overlay_df.set_index("date")[["temp_signal_c"]], height=260)
            except Exception as e:
                st.error(f"Failed to read overlay CSV: {e}")

    # Run detection on simulated series
    sim_df = sim.copy()
    sim_df["date"] = pd.to_datetime(sim_df["date"]).dt.date
    # reuse detection by faking required columns
    sim_df["temp_signal"] = (sim_df["temp_signal_c"] - sim_df["temp_signal_c"].mean()) / (sim_df["temp_signal_c"].std(ddof=0) or 1.0)

    # Try inferring menses or use optional inputs
    starts = menses_starts if len(menses_starts) > 0 else infer_menses_starts(sim_df)
    labeled, cycles = label_phases(sim_df, starts, RISE_MIN, RISE_DAYS)

    st.markdown("#### Simulated Series — Labeled")
    st.dataframe(labeled, use_container_width=True, hide_index=True)

    c1, c2 = st.columns([2,1])
    with c1:
        st.line_chart(labeled.set_index(pd.to_datetime(labeled["date"]))[["temp_signal_c"]], height=280)
    with c2:
        st.markdown("#### Detected Cycles (Sim)")
        if cycles:
            for cyc in cycles:
                st.write(f"- **Cycle start:** {cyc['cycle_start']}  \n"
                         f"  **Ovulation:** {cyc.get('ovulation', 'not found')}  \n"
                         f"  **Confidence:** {cyc.get('confidence', '—')}")
        else:
            st.write("No cycles summarized.")

    # Optional overlay comparison stats
    if overlay_df is not None:
        merged = pd.merge(
            labeled.rename(columns={"temp_signal_c": "sim_temp_c"}),
            overlay_df.rename(columns={"temp_signal_c": "oura_temp_c"}),
            on="date", how="inner"
        )
        if not merged.empty:
            st.markdown("#### Comparison (Sim vs Oura)")
            st.write(f"Days overlapped: **{len(merged)}**")
            corr = float(np.corrcoef(merged["sim_temp_c"], merged["oura_temp_c"])[0,1])
            st.write(f"Pearson correlation of temperatures: **{corr:.3f}**")

# ----------------------
# Main switch
# ----------------------
if mode == "Oura CSV Upload":
    oura_csv_ui()
else:
    sensor_stream_ui()

st.markdown("---")
with st.expander("README / Setup"):
    st.markdown(
        """
        **Install & Run**
        ```bash
        pip install -r requirements.txt
        streamlit run app.py
        ```

        **CSV Columns Accepted**
        - Date: `date`, `summary_date`, or `day`
        - Temperature (prefer **trend**): `temp_trend_deviation`, `temperature_trend_deviation`, `temp_trend`
        - Or raw deviation: `temperature_deviation`, `temp_deviation`
        - Optional helpers: `resting_heart_rate` (or `rhr`), `hrv` (`rmssd`), `respiratory_rate`

        **How detection works (TL;DR)**
        - Find a **nadir** followed by a **sustained rise** ≥ *RISE_MIN* for *RISE_DAYS*.
        - Ovulation ≈ day after the nadir; label phases around that.
        - Confidence increases with stronger, cleaner rises and optional RHR↑ / HRV↓ corroboration.

        **Sensor Mode**
        - The app simulates a daily BBT series. For real hardware, swap in a `pyserial` loop that reads `date,temperature_c` lines and appends to the dataframe.
        - Keep sampling frequency consistent (e.g., nightly min temp or a nightly aggregate).

        **Export**
        - Use the download button to save your labeled CSV with per-day phase and ovulation estimate.
        """
    )
