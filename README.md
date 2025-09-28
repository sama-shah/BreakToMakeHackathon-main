# Cycle Phase & Ovulation Detector (Streamlit)

A drag-and-drop Streamlit app that labels cycle phases and estimates ovulation from **Oura nightly temperature** exports — and lets you **simulate a BBT-like sensor stream** for hackathon demos.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

- **CSV mode:** Upload an Oura CSV with `date` and either `temp_trend_deviation` (preferred) or `temperature_deviation`. Optional helpers: `resting_heart_rate`, `hrv` (RMSSD), `respiratory_rate`.
- **Sensor mode:** Uses a simulated BBT series. Replace the simulation with a `pyserial` loop for your Arduino 10k NTC build if desired.

## Detection
- Detects nadir → sustained rise (≥0.25°C for ≥3 days by default). Ovulation ≈ day after nadir.
- Labels phases: menstruation, follicular, ovulation_window, luteal, premenstrual.
- Confidence blends rise magnitude/persistence and optional RHR↑ / HRV↓ corroboration.
