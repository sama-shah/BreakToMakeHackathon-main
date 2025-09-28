
import streamlit as st
import pandas as pd
import numpy as np
import time, json, sys
from datetime import timedelta, datetime
import altair as alt

# Optional: only import pyserial on the Live tab to avoid errors if it's not installed yet.
try:
    import serial
except Exception:
    serial = None

st.set_page_config(page_title="Oura Timeline + Live Sensor Demo", layout="wide")
st.title("Oura Timeline + Live Sensor Demo")

tab_replay, tab_live = st.tabs(["📈 Oura timeline (120‑day replay)", "🛠️ Live sensor (Arduino)"])

# =============================
# Shared helpers
# =============================
def extract_trend_first(df: pd.DataFrame, smooth_win: int) -> pd.Series:
    lc_map = {c.lower().strip(): c for c in df.columns}
    for key in ["temperature_trend_deviation", "temp_trend_deviation", "temp_trend"]:
        if key in lc_map:
            return pd.to_numeric(df[lc_map[key]], errors="coerce")
    for lc_name, orig in lc_map.items():
        if "temperature_trend_deviation" in lc_name or "temp_trend_deviation" in lc_name:
            return pd.to_numeric(df[orig], errors="coerce")
    for rc in [c for c in df.columns if c.lower().strip() in ["readiness", "readiness_json", "readiness_data"]]:
        try:
            if isinstance(df[rc].iloc[0], dict):
                series = df[rc].apply(lambda x: x.get("contributors", {}).get("temperature_trend_deviation", np.nan) if isinstance(x, dict) else np.nan)
                if series.notna().sum() > 0:
                    return pd.to_numeric(series, errors="coerce")
            else:
                parsed = df[rc].apply(lambda s: json.loads(s) if isinstance(s, str) and s.strip().startswith("{") else {})
                series = parsed.apply(lambda x: x.get("contributors", {}).get("temperature_trend_deviation", np.nan) if isinstance(x, dict) else np.nan)
                if series.notna().sum() > 0:
                    return pd.to_numeric(series, errors="coerce")
        except Exception:
            pass
    for key in ["temperature_deviation", "temp_deviation"]:
        if key in lc_map:
            s = pd.to_numeric(df[lc_map[key]], errors="coerce")
            return s.rolling(5, min_periods=1, center=True).mean()
    for lc_name, orig in lc_map.items():
        if "temperature_deviation" in lc_name:
            s = pd.to_numeric(df[orig], errors="coerce")
            return s.rolling(5, min_periods=1, center=True).mean()
    raise ValueError("No temperature trend/deviation column found.")

def prepare_df(raw: pd.DataFrame, smooth_win: int) -> pd.DataFrame:
    date_col = None
    for c in raw.columns:
        if c.lower().strip() in ["date","summary_date","day"] or "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        raise ValueError("CSV must include a date-like column (date/summary_date/day).")
    df = raw.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").drop_duplicates("date")
    temp = extract_trend_first(df, 5)
    df["temp_signal_c"] = temp
    std = float(temp.std(ddof=0)) if float(temp.std(ddof=0)) > 0 else 1.0
    df["temp_signal"] = (temp - float(temp.mean())) / std
    for opt, cands in {
        "rhr": ["resting_heart_rate","rhr","average_resting_heart_rate"],
        "hrv": ["hrv","rmssd"],
        "resp_rate": ["respiratory_rate","breath_average"]
    }.items():
        for c in raw.columns:
            if c.lower().strip() in cands:
                df = df.rename(columns={c: opt})
                df[opt] = pd.to_numeric(df[opt], errors="coerce").interpolate(limit_direction="both")
                break
    return df

def find_nadir_then_rise(sig_c, idx, rise_min=0.25, rise_days=3):
    n = len(sig_c)
    if idx + rise_days >= n:
        return False, None, 0.0
    base = sig_c.iloc[idx]
    window = sig_c.iloc[idx+1: idx+1+rise_days]
    cond = (window - base) >= rise_min
    if cond.all():
        return True, idx+1, float((window - base).mean())
    return False, None, float((window - base).clip(lower=0).mean())

def detect_ovulation_candidates(df: pd.DataFrame, rise_min=0.25, rise_days=3, search_start=None, search_end=None):
    sig_c = df["temp_signal_c"].reset_index(drop=True)
    dates = pd.Series(df["date"]).reset_index(drop=True)
    candidates = []
    for i in range(1, len(sig_c)-1):
        if search_start and dates[i] < search_start: 
            continue
        if search_end and dates[i] > search_end:
            continue
        if sig_c.iloc[i] < sig_c.iloc[i-1] and sig_c.iloc[i] < sig_c.iloc[i+1]:
            ok, rise_idx, mean_rise = find_nadir_then_rise(sig_c, i, rise_min, rise_days)
            if ok:
                rhr_delta = hrv_delta = np.nan
                if "rhr" in df.columns:
                    rhr_delta = (df["rhr"].iloc[min(rise_idx+3, len(df)-1)] - df["rhr"].iloc[i:i+1].mean())
                if "hrv" in df.columns:
                    hrv_delta = (df["hrv"].iloc[min(rise_idx+3, len(df)-1)] - df["hrv"].iloc[i:i+1].mean())
                candidates.append({
                    "ovulation_date": dates[rise_idx],
                    "nadir_date": dates[i],
                    "rise_mean_c": float(mean_rise),
                    "nadir_temp_c": float(sig_c.iloc[i]),
                    "support_rhr_delta": float(rhr_delta) if pd.notna(rhr_delta) else None,
                    "support_hrv_delta": float(hrv_delta) if pd.notna(hrv_delta) else None
                })
    return candidates

def choose_best_candidate(cands):
    if not cands:
        return None
    def score(c):
        s = c["rise_mean_c"]
        if c["support_rhr_delta"] is not None:
            s += 0.05 * np.sign(c["support_rhr_delta"])
        if c["support_hrv_delta"] is not None:
            s += 0.05 * (-np.sign(c["support_hrv_delta"]))
        return s
    return max(cands, key=score)

def infer_menses_starts(df, min_cycle_len=21, max_cycle_len=45):
    temp = df["temp_signal_c"]
    mean = temp.rolling(7, min_periods=3).mean()
    dips = (temp < (mean - 0.05)) & (temp.diff() < -0.1)
    dates = df["date"]
    starts = []
    prev = None
    for i, is_dip in enumerate(dips):
        if not is_dip: continue
        d = dates.iloc[i]
        if prev and (d - prev).days < min_cycle_len:
            continue
        starts.append(d)
        prev = d
    return starts

def label_phases(df, menses_starts, rise_min=0.25, rise_days=3):
    out = df.copy()
    out["phase"] = "unlabeled"
    out["ovulation_estimate"] = pd.NaT
    out["confidence"] = np.nan
    dates = out["date"]
    if not menses_starts:
        cands = detect_ovulation_candidates(out, rise_min, rise_days)
        best = choose_best_candidate(cands)
        if best:
            ovu = best["ovulation_date"]
            out.loc[dates.between(ovu - timedelta(days=1), ovu + timedelta(days=1)), "phase"] = "ovulation_window"
            out.loc[dates < ovu - timedelta(days=1), "phase"] = "follicular"
            out.loc[dates > ovu + timedelta(days=1), "phase"] = "luteal"
            out.loc[dates == ovu, "ovulation_estimate"] = pd.to_datetime(ovu)
            out["confidence"] = best["rise_mean_c"] / 0.5
        return out
    for i, start in enumerate(menses_starts):
        end = menses_starts[i+1] - timedelta(days=1) if i+1 < len(menses_starts) else dates.max()
        cyc_mask = dates.between(start, end)
        if not cyc_mask.any(): 
            continue
        for d in range(5):
            day = start + timedelta(days=d)
            if day > end: break
            out.loc[out["date"] == day, "phase"] = "menstruation"
        win_start = max(start + timedelta(days=8), start)
        win_end = min(start + timedelta(days=20), end)
        cands = detect_ovulation_candidates(out.loc[cyc_mask], rise_min, rise_days, search_start=win_start, search_end=win_end)
        best = choose_best_candidate(cands)
        if best:
            ovu = best["ovulation_date"]
            out.loc[out["date"].between(ovu - timedelta(days=1), ovu + timedelta(days=1)), "phase"] = "ovulation_window"
            out.loc[out["date"] == ovu, "ovulation_estimate"] = pd.to_datetime(ovu)
            out.loc[out["date"].between(start + timedelta(days=5), (ovu - timedelta(days=2))), "phase"] = "follicular"
            out.loc[out["date"].between(ovu + timedelta(days=2), end), "phase"] = "luteal"
            idx = menses_starts.index(start)
            if idx+1 < len(menses_starts):
                next_start = menses_starts[idx+1]
                premsk = out["date"].between(next_start - timedelta(days=3), next_start - timedelta(days=1))
                out.loc[premsk, "phase"] = "premenstrual"
            conf = max(0.0, min(1.0, best["rise_mean_c"] / 0.5))
            out.loc[cyc_mask, "confidence"] = out.loc[cyc_mask, "confidence"].fillna(conf)
        else:
            mid = out.loc[cyc_mask, "temp_signal_c"].median()
            lowmask = cyc_mask & (out["temp_signal_c"] <= mid)
            highmask = cyc_mask & (out["temp_signal_c"] > mid)
            out.loc[lowmask & (out["phase"] == "unlabeled"), "phase"] = "follicular"
            out.loc[highmask & (out["phase"] == "unlabeled"), "phase"] = "luteal"
    return out

# =============================
# TAB 1: Oura timeline replay
# =============================
with tab_replay:
    st.subheader("Upload Oura CSV (trend preferred)")
    c1, c2 = st.columns([3,2])
    with c2:
        st.markdown("**Playback**: 1 day/sec → 30 seconds plays 30 days")
        speed = st.slider("Days per second", 0.25, 5.0, 1.0, 0.25, key='replay_speed')
        RISE_MIN = st.slider("Rise Threshold (°C)", 0.1, 0.6, 0.25, 0.01, key='replay_rise')
        RISE_DAYS = st.slider("Rise Must Persist (days)", 2, 5, 3, 1, key='replay_days')
        expected_min_cycle = st.slider("Min cycle length", 18, 26, 21, 1, key='replay_mincl')
        expected_max_cycle = st.slider("Max cycle length", 30, 60, 45, 1, key='replay_maxcl')
    with c1:
        upl = st.file_uploader("Drop Oura CSV", type=["csv"], key='replay_upl')

    if upl is None:
        st.info("Upload a CSV to start the replay.")
    else:
        try:
            raw = pd.read_csv(upl)
            full_df = prepare_df(raw, 5)
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")
            st.stop()

        first_date = full_df["date"].min()
        df120 = full_df[full_df["date"] >= first_date].head(120).reset_index(drop=True)
        if len(df120) < 10:
            st.warning("Need at least ~10 days of data for a meaningful demo.")
        # state
        if "play_state" not in st.session_state:
            st.session_state.play_state = {"playing": False, "start_ts": None, "frame_idx": 0, "speed": speed}
        state = st.session_state.play_state
        # controls
        bcol1, bcol2, bcol3, bcol4 = st.columns([1,1,1,2])
        with bcol1:
            if st.button("▶️ Play", use_container_width=True, key='replay_play'):
                state["playing"] = True
                if state["start_ts"] is None:
                    state["start_ts"] = time.time()
        with bcol2:
            if st.button("⏸️ Pause", use_container_width=True, key='replay_pause'):
                state["playing"] = False
        with bcol3:
            if st.button("⟲ Reset", use_container_width=True, key='replay_reset'):
                state.update({"playing": False, "start_ts": None, "frame_idx": 0})
        with bcol4:
            state["frame_idx"] = st.slider("Scrub day (0–119)", 0, max(0, len(df120)-1), state["frame_idx"], key='replay_scrub')

        # adapt speed
        if state["playing"] and state["speed"] != speed:
            now = time.time()
            state["start_ts"] = now - (state["frame_idx"] / speed)
            state["speed"] = speed

        if state["playing"]:
            elapsed = 0.0 if state["start_ts"] is None else (time.time() - state["start_ts"])
            next_idx = int(elapsed * speed)
            state["frame_idx"] = min(next_idx, len(df120)-1)
            st.experimental_rerun()

        cur_idx = state["frame_idx"]
        cur_df = df120.iloc[:cur_idx+1].copy()
        menses_starts = infer_menses_starts(cur_df, expected_min_cycle, expected_max_cycle)
        labeled = label_phases(cur_df, menses_starts, RISE_MIN, RISE_DAYS)

        cur_date = labeled["date"].iloc[-1]
        cur_phase = labeled["phase"].iloc[-1]
        cur_ovu = labeled.loc[labeled["ovulation_estimate"].notna(), "ovulation_estimate"]
        cur_ovu_display = pd.to_datetime(cur_ovu.iloc[-1]).date() if not cur_ovu.empty else "—"

        st.subheader("Live status")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Day", f"{cur_idx+1} / {len(df120)}")
        m2.metric("Date", f"{cur_date}")
        m3.metric("Phase", f"{cur_phase}")
        m4.metric("Last ovulation", f"{cur_ovu_display}")

        plot_df = labeled.copy()
        plot_df["date"] = pd.to_datetime(plot_df["date"])
        phase_color = alt.Scale(domain=["menstruation","follicular","ovulation_window","luteal","premenstrual","unlabeled"],
                                range=["#e74c3c","#1abc9c","#f1c40f","#9b59b6","#e67e22","#95a5a6"])
        line = alt.Chart(plot_df).mark_line().encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('temp_signal_c:Q', title='Temperature deviation (°C vs baseline)'),
            color=alt.Color('phase:N', scale=phase_color, legend=alt.Legend(title="Phase"))
        )
        cursor = alt.Chart(pd.DataFrame({"date":[plot_df["date"].iloc[-1]]})).mark_rule(strokeDash=[4,4]).encode(x='date:T')
        points = alt.Chart(plot_df).mark_point(size=28, filled=True).encode(
            x='date:T', y='temp_signal_c:Q', color=alt.Color('phase:N', scale=phase_color, legend=None)
        )
        st.altair_chart((line + points + cursor).interactive().properties(height=340), use_container_width=True)
        st.dataframe(labeled.tail(10), use_container_width=True, hide_index=True)

# =============================
# TAB 2: Live Arduino stream
# =============================
with tab_live:
    st.subheader("Connect Arduino (Temp, HR, HRV)")
    st.markdown(r"""
**Serial format (CSV)** expected by default:
```
2025-09-28T12:34:56Z, 36.45, 72, 830
# iso_datetime, temp_c, bpm, ibi_ms
```
Or **JSON**:
```json
{"t":"2025-09-28T12:34:56Z","temp_c":36.45,"bpm":72,"ibi_ms":830}
```
""")
    sim = st.checkbox("Use simulation (no hardware)", value=False, help="Great for dry runs at the hackathon booth.")
    port = st.text_input("Serial port (e.g., COM3 on Windows, /dev/tty.usbmodemXXXX on macOS)", value="")
    baud = st.number_input("Baud rate", 300, 1000000, 115200, step=300)
    fmt = st.selectbox("Stream format", ["CSV (iso,temp_c,bpm,ibi_ms)", "JSON ({t,temp_c,bpm,ibi_ms})"])
    run_secs = st.slider("Capture duration (seconds)", 5, 180, 30, 5)
    clear_btn = st.button("🧹 Clear buffer")

    if "live_df" not in st.session_state or clear_btn:
        st.session_state.live_df = pd.DataFrame(columns=["ts","temp_c","bpm","ibi_ms"])

    run = st.button("▶️ Start capture", type="primary")
    status = st.empty()
    chart_placeholder_temp = st.empty()
    chart_placeholder_bpm = st.empty()
    chart_placeholder_ibi = st.empty()
    table_placeholder = st.empty()
    metrics = st.columns(3)

    def rmssd_ms(ibi_ms_series: pd.Series) -> float:
        if len(ibi_ms_series) < 3:
            return np.nan
        vals = ibi_ms_series.dropna().values
        if len(vals) < 3:
            return np.nan
        diff = np.diff(vals)
        if len(diff) == 0:
            return np.nan
        return float(np.sqrt(np.mean(diff**2)))

    def parse_line(line: str):
        line = line.strip()
        if not line:
            return None
        if fmt.startswith("JSON"):
            try:
                o = json.loads(line)
                t = o.get("t") or o.get("ts") or o.get("time")
                temp_c = float(o.get("temp_c")) if o.get("temp_c") is not None else np.nan
                bpm = float(o.get("bpm")) if o.get("bpm") is not None else np.nan
                ibi = float(o.get("ibi_ms")) if o.get("ibi_ms") is not None else np.nan
                ts = pd.to_datetime(t, errors="coerce")
                if pd.isna(ts):
                    ts = pd.Timestamp.utcnow()
                return ts, temp_c, bpm, ibi
            except Exception:
                return None
        else:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                return None
            ts = pd.to_datetime(parts[0], errors="coerce")
            if pd.isna(ts):
                ts = pd.Timestamp.utcnow()
            def to_f(x):
                try: return float(x)
                except: return np.nan
            temp_c = to_f(parts[1])
            bpm = to_f(parts[2])
            ibi = to_f(parts[3])
            return ts, temp_c, bpm, ibi

    def simulate_sample(i: int):
        ts = pd.Timestamp.utcnow()
        temp_c = 36.3 + 0.3*np.sin((i/60.0)*2*np.pi) + 0.05*np.random.randn()
        bpm = 74 + 6*np.sin((i/10.0)*2*np.pi) + 1.5*np.random.randn()
        ibi = 60000.0 / max(40.0, bpm + np.random.randn())
        return ts, float(temp_c), float(bpm), float(ibi)

    if run:
        ser = None
        if not sim:
            if serial is None:
                st.error("pyserial not installed. Run: `pip install pyserial`")
                st.stop()
            if not port:
                st.error("Please enter a serial port or use simulation.")
                st.stop()
            try:
                ser = serial.Serial(port, int(baud), timeout=1)
                status.info(f"Connected to **{port}** @ {baud} baud")
            except Exception as e:
                st.error(f"Failed to open serial port: {e}")
                st.stop()
        else:
            status.info("Simulation mode active. No hardware required.")

        t0 = time.time()
        i = 0
        while time.time() - t0 < run_secs:
            if sim:
                sample = simulate_sample(i)
            else:
                try:
                    raw = ser.readline().decode(errors="ignore")
                except Exception:
                    raw = ""
                sample = parse_line(raw)
                if sample is None:
                    time.sleep(0.02)
                    i += 1
                    continue

            ts, temp_c, bpm, ibi = sample
            new_row = {"ts": ts, "temp_c": temp_c, "bpm": bpm, "ibi_ms": ibi}
            st.session_state.live_df = pd.concat([st.session_state.live_df, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state.live_df = st.session_state.live_df.sort_values("ts").tail(3600)

            df = st.session_state.live_df.copy()
            recent = df[df["ts"] >= (pd.Timestamp.utcnow() - pd.Timedelta(seconds=60))]
            hrv_rmssd = rmssd_ms(recent["ibi_ms"]) if len(recent) >= 3 else np.nan

            plot_df = df.tail(600).copy()
            plot_df["ts"] = pd.to_datetime(plot_df["ts"])

            temp_chart = alt.Chart(plot_df).mark_line().encode(
                x=alt.X('ts:T', title='Time'),
                y=alt.Y('temp_c:Q', title='Temp (°C)')
            ).properties(height=160)

            bpm_chart = alt.Chart(plot_df).mark_line().encode(
                x=alt.X('ts:T', title='Time'),
                y=alt.Y('bpm:Q', title='BPM')
            ).properties(height=160)

            ibi_chart = alt.Chart(plot_df).mark_line().encode(
                x=alt.X('ts:T', title='Time'),
                y=alt.Y('ibi_ms:Q', title='IBI (ms)')
            ).properties(height=160)

            chart_placeholder_temp.altair_chart(temp_chart, use_container_width=True)
            chart_placeholder_bpm.altair_chart(bpm_chart, use_container_width=True)
            chart_placeholder_ibi.altair_chart(ibi_chart, use_container_width=True)

            metrics[0].metric("Temp (°C)", f"{temp_c:.2f}" if not np.isnan(temp_c) else "—")
            metrics[1].metric("BPM", f"{bpm:.0f}" if not np.isnan(bpm) else "—")
            metrics[2].metric("HRV (RMSSD, 60s, ms)", f"{hrv_rmssd:.0f}" if not np.isnan(hrv_rmssd) else "—")

            table_placeholder.dataframe(df.tail(10), use_container_width=True, hide_index=True)
            time.sleep(0.05)
            i += 1

        status.success("Capture complete.")
        if not sim and ser is not None:
            try:
                ser.close()
            except Exception:
                pass

st.caption("Note: This demo is not a medical device. For personal health decisions, consult a clinician.")
