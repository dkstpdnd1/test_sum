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


## V2 라우터 수정
- Streamlit Cloud Main file path가 `app.py`여도 동작합니다.
- Main file path가 기존 설정처럼 `pages/app.py`여도 동작합니다.
- Streamlit 1.37의 `st.Page` 상대경로 규칙에 맞춰 진입 파일 위치별로 페이지 경로를 자동 선택합니다.
- 권장 Main file path는 저장소 루트의 `app.py`입니다.


## V3 중요 수정
- `pages/requirements.txt`를 제거했습니다. Streamlit Community Cloud는 진입 파일과 같은 폴더의 requirements.txt를 루트보다 우선 사용하므로, `pages/app.py`가 진입점일 때 기존 파일이 root requirements를 가려 01 페이지의 `altair`, `scipy`, `opencv-python-headless` 등이 설치되지 않았습니다.
- 이제 의존성 파일은 루트 `requirements.txt` 하나만 사용합니다.
- `pages/app.py`에서 프로젝트 루트를 `sys.path`에 명시적으로 추가해 `modules.*` import를 안정화했습니다.
