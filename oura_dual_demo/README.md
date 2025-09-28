# Oura Timeline + Live Sensor Demo

Two tabs:
1) **Oura timeline** (first 120 days, trend preferred)
2) **Live sensor** (Arduino serial → temp, BPM, IBI → HRV RMSSD)

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Serial formats
CSV: `ISO8601,temp_c,bpm,ibi_ms`
JSON: `{"t": "...", "temp_c": 36.5, "bpm": 70, "ibi_ms": 860}`

## Ports
- macOS `/dev/tty.usbmodem*` or `/dev/tty.usbserial*`
- Linux `/dev/ttyACM0` or `/dev/ttyUSB0`
- Windows `COM3` (check Device Manager)
