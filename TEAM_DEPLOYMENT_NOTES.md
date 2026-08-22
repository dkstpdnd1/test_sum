# Team deployment notes

## Environment
- Streamlit is pinned to `1.37.1` so Cloud rendering matches the tested local UI.
- Root `.streamlit/config.toml` fixes native widgets to the same dark/blue palette.
- `st.set_page_config()` is called only in root `app.py`; individual pages must not call it.

## Run locally
```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Streamlit Community Cloud
After pushing these files, use **Manage app → Reboot app** so the pinned Streamlit version and theme are rebuilt.

## Page 3 data
- `pages/data/operation_dashboard_oct2025.csv.gz`
- `pages/data/flight_counter_oct2025.csv`
- analysis window: 2025-09-01 through 2025-10-31

## Secrets
Do not commit real API keys. Use Streamlit Cloud Secrets or a local `.streamlit/secrets.toml`.
